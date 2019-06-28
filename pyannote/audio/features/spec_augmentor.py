import numpy as np
from pyannote.core import SlidingWindowFeature
import random
import torch


def time_warper(spec, W=5):
    num_rows = spec.shape[1]
    spec_len = spec.shape[2]
    device = spec.device

    y = num_rows // 2
    horizontal_line_at_ctr = spec[0][y]
    assert len(horizontal_line_at_ctr) == spec_len

    point_to_warp = horizontal_line_at_ctr[random.randrange(W, spec_len - W)]
    assert isinstance(point_to_warp, torch.Tensor)

    # Uniform distribution from (0,W) with chance to be up to W negative
    dist_to_warp = random.randrange(-W, W)
    src_pts, dest_pts = (torch.tensor([[[y, point_to_warp]]], device=device),
                         torch.tensor([[[y, point_to_warp + dist_to_warp]]], device=device))
    warped_spectro, dense_flows = sparse_image_warp(spec, src_pts, dest_pts)
    return warped_spectro.squeeze(3)

def sparse_image_warp(img_tensor,
                      source_control_point_locations,
                      dest_control_point_locations,
                      interpolation_order=2,
                      regularization_weight=0.0,
                      num_boundaries_points=0):
    control_point_flows = (dest_control_point_locations - source_control_point_locations)

#     clamp_boundaries = num_boundary_points > 0
#     boundary_points_per_edge = num_boundary_points - 1
    batch_size, image_height, image_width = img_tensor.shape
    grid_locations = get_grid_locations(image_height, image_width)
    flattened_grid_locations = torch.tensor(flatten_grid_locations(grid_locations, image_height, image_width))

#     flattened_grid_locations = constant_op.constant(
#         _expand_to_minibatch(flattened_grid_locations, batch_size), image.dtype)

#     if clamp_boundaries:
#       (dest_control_point_locations,
#        control_point_flows) = _add_zero_flow_controls_at_boundary(
#            dest_control_point_locations, control_point_flows, image_height,
#            image_width, boundary_points_per_edge)

    flattened_flows = interpolate_spline(
        dest_control_point_locations,
        control_point_flows,
        flattened_grid_locations,
        interpolation_order,
        regularization_weight)

    dense_flows = create_dense_flows(flattened_flows, batch_size, image_height, image_width)

    warped_image = dense_image_warp(img_tensor, dense_flows)

    return warped_image, dense_flows

def get_grid_locations(image_height, image_width):
    """Wrapper for np.meshgrid."""

    y_range = np.linspace(0, image_height - 1, image_height)
    x_range = np.linspace(0, image_width - 1, image_width)
    y_grid, x_grid = np.meshgrid(y_range, x_range, indexing='ij')
    return np.stack((y_grid, x_grid), -1)

def flatten_grid_locations(grid_locations, image_height, image_width):
    return np.reshape(grid_locations, [image_height * image_width, 2])

def create_dense_flows(flattened_flows, batch_size, image_height, image_width):
    # possibly .view
    return torch.reshape(flattened_flows, [batch_size, image_height, image_width, 2])

def interpolate_spline(train_points, train_values, query_points, order, regularization_weight=0.0,):
    # First, fit the spline to the observed data.
    w, v = solve_interpolation(train_points, train_values, order, regularization_weight)
    # Then, evaluate the spline at the query locations.
    query_values = apply_interpolation(query_points, train_points, w, v, order)

    return query_values

def solve_interpolation(train_points, train_values, order, regularization_weight):
    b, n, d = train_points.shape
    k = train_values.shape[-1]

    # First, rename variables so that the notation (c, f, w, v, A, B, etc.)
    # follows https://en.wikipedia.org/wiki/Polyharmonic_spline.
    # To account for python style guidelines we use
    # matrix_a for A and matrix_b for B.

    c = train_points
    f = train_values.float()

    matrix_a = phi(cross_squared_distance_matrix(c,c), order).unsqueeze(0)  # [b, n, n]
#     print('Matrix A', matrix_a, matrix_a.shape)
#     if regularization_weight > 0:
#         batch_identity_matrix = array_ops.expand_dims(
#           linalg_ops.eye(n, dtype=c.dtype), 0)
#         matrix_a += regularization_weight * batch_identity_matrix

    # Append ones to the feature values for the bias term in the linear model.
    ones = torch.ones(1, dtype=train_points.dtype).view([-1, 1, 1])
    matrix_b = torch.cat((c, ones), 2).float()  # [b, n, d + 1]
#     print('Matrix B', matrix_b, matrix_b.shape)

    # [b, n + d + 1, n]
    left_block = torch.cat((matrix_a, torch.transpose(matrix_b, 2, 1)), 1)
#     print('Left Block', left_block, left_block.shape)

    num_b_cols = matrix_b.shape[2]  # d + 1
#     print('Num_B_Cols', matrix_b.shape)
    #     lhs_zeros = torch.zeros((b, num_b_cols, num_b_cols), dtype=train_points.dtype).float()

    # In Tensorflow, zeros are used here. Pytorch gesv fails with zeros for some reason we don't understand.
    # So instead we use very tiny randn values (variance of one, zero mean) on one side of our multiplication.
    lhs_zeros = torch.randn((b, num_b_cols, num_b_cols)) / 1e10
    right_block = torch.cat((matrix_b, lhs_zeros),
                                   1)  # [b, n + d + 1, d + 1]
#     print('Right Block', right_block, right_block.shape)
    lhs = torch.cat((left_block, right_block),
                           2)  # [b, n + d + 1, n + d + 1]
#     print('LHS', lhs, lhs.shape)

    rhs_zeros = torch.zeros((b, d + 1, k), dtype=train_points.dtype).float()
    rhs = torch.cat((f, rhs_zeros), 1)  # [b, n + d + 1, k]
#     print('RHS', rhs, rhs.shape)

    # Then, solve the linear system and unpack the results.
    X, LU = torch.solve(rhs, lhs)
    w = X[:, :n, :]
    v = X[:, n:, :]

    return w, v

def cross_squared_distance_matrix(x, y):
    """Pairwise squared distance between two (batch) matrices' rows (2nd dim).
        Computes the pairwise distances between rows of x and rows of y
        Args:
        x: [batch_size, n, d] float `Tensor`
        y: [batch_size, m, d] float `Tensor`
        Returns:
        squared_dists: [batch_size, n, m] float `Tensor`, where
        squared_dists[b,i,j] = ||x[b,i,:] - y[b,j,:]||^2
    """
    x_norm_squared = torch.sum(torch.mul(x, x))
    y_norm_squared = torch.sum(torch.mul(y, y))

    x_y_transpose = torch.matmul(x.squeeze(0), y.squeeze(0).transpose(0,1))

    # squared_dists[b,i,j] = ||x_bi - y_bj||^2 = x_bi'x_bi- 2x_bi'x_bj + x_bj'x_bj
    squared_dists = x_norm_squared - 2 * x_y_transpose + y_norm_squared

    return squared_dists.float()

def phi(r, order):
    """Coordinate-wise nonlinearity used to define the order of the interpolation.
    See https://en.wikipedia.org/wiki/Polyharmonic_spline for the definition.
    Args:
    r: input op
    order: interpolation order
    Returns:
    phi_k evaluated coordinate-wise on r, for k = r
    """
    EPSILON=torch.tensor(1e-10)
    # using EPSILON prevents log(0), sqrt0), etc.
    # sqrt(0) is well-defined, but its gradient is not
    if order == 1:
        r = torch.max(r, EPSILON)
        r = torch.sqrt(r)
        return r
    elif order == 2:
        return 0.5 * r * torch.log(torch.max(r, EPSILON))
    elif order == 4:
        return 0.5 * torch.square(r) * torch.log(torch.max(r, EPSILON))
    elif order % 2 == 0:
        r = torch.max(r, EPSILON)
        return 0.5 * torch.pow(r, 0.5 * order) * torch.log(r)
    else:
        r = torch.max(r, EPSILON)
        return torch.pow(r, 0.5 * order)

def apply_interpolation(query_points, train_points, w, v, order):
    """Apply polyharmonic interpolation model to data.
    Given coefficients w and v for the interpolation model, we evaluate
    interpolated function values at query_points.
    Args:
    query_points: `[b, m, d]` x values to evaluate the interpolation at
    train_points: `[b, n, d]` x values that act as the interpolation centers
                    ( the c variables in the wikipedia article)
    w: `[b, n, k]` weights on each interpolation center
    v: `[b, d, k]` weights on each input dimension
    order: order of the interpolation
    Returns:
    Polyharmonic interpolation evaluated at points defined in query_points.
    """
    query_points = query_points.unsqueeze(0)
    # First, compute the contribution from the rbf term.
#     print(query_points.shape, train_points.shape)
    pairwise_dists = cross_squared_distance_matrix(query_points.float(), train_points.float())
#     print('Pairwise', pairwise_dists)
    phi_pairwise_dists = phi(pairwise_dists, order)
#     print('Pairwise phi', phi_pairwise_dists)

    rbf_term = torch.matmul(phi_pairwise_dists, w)

    # Then, compute the contribution from the linear term.
    # Pad query_points with ones, for the bias term in the linear model.
    ones = torch.ones_like(query_points[..., :1])
    query_points_pad = torch.cat((
      query_points,
      ones
    ), 2).float()
    linear_term = torch.matmul(query_points_pad, v)

    return rbf_term + linear_term


def dense_image_warp(image, flow):
    """Image warping using per-pixel flow vectors.
    Apply a non-linear warp to the image, where the warp is specified by a dense
    flow field of offset vectors that define the correspondences of pixel values
    in the output image back to locations in the  source image. Specifically, the
    pixel value at output[b, j, i, c] is
    images[b, j - flow[b, j, i, 0], i - flow[b, j, i, 1], c].
    The locations specified by this formula do not necessarily map to an int
    index. Therefore, the pixel value is obtained by bilinear
    interpolation of the 4 nearest pixels around
    (b, j - flow[b, j, i, 0], i - flow[b, j, i, 1]). For locations outside
    of the image, we use the nearest pixel values at the image boundary.
    Args:
    image: 4-D float `Tensor` with shape `[batch, height, width, channels]`.
    flow: A 4-D float `Tensor` with shape `[batch, height, width, 2]`.
    name: A name for the operation (optional).
    Note that image and flow can be of type tf.half, tf.float32, or tf.float64,
    and do not necessarily have to be the same type.
    Returns:
    A 4-D float `Tensor` with shape`[batch, height, width, channels]`
    and same type as input image.
    Raises:
    ValueError: if height < 2 or width < 2 or the inputs have the wrong number
    of dimensions.
    """
    image = image.unsqueeze(3) # add a single channel dimension to image tensor
    batch_size, height, width, channels = image.shape

    # The flow is defined on the image grid. Turn the flow into a list of query
    # points in the grid space.
    grid_x, grid_y = torch.meshgrid(
        torch.arange(width), torch.arange(height))

    stacked_grid = torch.stack((grid_y, grid_x), dim=2).float()
#     print('stacked', stacked_grid.shape)

    batched_grid = stacked_grid.unsqueeze(-1).permute(3, 1, 0, 2)
#     print('batched_grid', batched_grid.shape)

    query_points_on_grid = batched_grid - flow
    query_points_flattened = torch.reshape(query_points_on_grid,
                                               [batch_size, height * width, 2])
    # Compute values at the query points, then reshape the result back to the
    # image grid.
#     print('Query points', query_points_flattened, query_points_flattened.shape)
    interpolated = interpolate_bilinear(image, query_points_flattened)
    interpolated = torch.reshape(interpolated,
                                     [batch_size, height, width, channels])
    return interpolated

def interpolate_bilinear(grid,
                         query_points,
                         name='interpolate_bilinear',
                         indexing='ij'):
    """Similar to Matlab's interp2 function.
    Finds values for query points on a grid using bilinear interpolation.
    Args:
    grid: a 4-D float `Tensor` of shape `[batch, height, width, channels]`.
    query_points: a 3-D float `Tensor` of N points with shape `[batch, N, 2]`.
    name: a name for the operation (optional).
    indexing: whether the query points are specified as row and column (ij),
      or Cartesian coordinates (xy).
    Returns:
    values: a 3-D `Tensor` with shape `[batch, N, channels]`
    Raises:
    ValueError: if the indexing mode is invalid, or if the shape of the inputs
      invalid.
    """
    if indexing != 'ij' and indexing != 'xy':
        raise ValueError('Indexing mode must be \'ij\' or \'xy\'')


    shape = grid.shape
    if len(shape) != 4:
      msg = 'Grid must be 4 dimensional. Received size: '
      raise ValueError(msg + str(grid.shape))

    batch_size, height, width, channels = grid.shape

    shape = [batch_size, height, width, channels]
    query_type = query_points.dtype
    grid_type = grid.dtype

    num_queries = query_points.shape[1]
#     print('Num queries', num_queries)

    alphas = []
    floors = []
    ceils = []
    index_order = [0, 1] if indexing == 'ij' else [1, 0]
#     print(query_points.shape)
    unstacked_query_points = query_points.unbind(2)
#     print('Squeezed query_points', unstacked_query_points[0].shape, unstacked_query_points[1].shape)

    for dim in index_order:
        queries = unstacked_query_points[dim]

        size_in_indexing_dimension = shape[dim + 1]

        # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
        # is still a valid index into the grid.
        max_floor = torch.tensor(size_in_indexing_dimension - 2, dtype=query_type)
        min_floor = torch.tensor(0.0, dtype=query_type)
        maxx = torch.max(min_floor, torch.floor(queries))
        floor = torch.min(maxx, max_floor)
        int_floor = floor.long()
        floors.append(int_floor)
        ceil = int_floor + 1
        ceils.append(ceil)

        # alpha has the same type as the grid, as we will directly use alpha
        # when taking linear combinations of pixel values from the image.
        alpha = (queries - floor).clone().detach()
        min_alpha = torch.tensor(0.0, dtype=grid_type)
        max_alpha = torch.tensor(1.0, dtype=grid_type)
        alpha = torch.min(torch.max(min_alpha, alpha), max_alpha)

        # Expand alpha to [b, n, 1] so we can use broadcasting
        # (since the alpha values don't depend on the channel).
        alpha = torch.unsqueeze(alpha, 2)
        alphas.append(alpha)

    flattened_grid = torch.reshape(
      grid, [batch_size * height * width, channels])
    batch_offsets = torch.reshape(
      torch.arange(batch_size) * height * width, [batch_size, 1])

    # This wraps array_ops.gather. We reshape the image data such that the
    # batch, y, and x coordinates are pulled into the first dimension.
    # Then we gather. Finally, we reshape the output back. It's possible this
    # code would be made simpler by using array_ops.gather_nd.
    def gather(y_coords, x_coords, name):
        linear_coordinates = batch_offsets + y_coords * width + x_coords
        gathered_values = torch.gather(flattened_grid.t(), 1, linear_coordinates)
        return torch.reshape(gathered_values,
                                 [batch_size, num_queries, channels])

    # grab the pixel values in the 4 corners around each query point
    top_left = gather(floors[0], floors[1], 'top_left')
    top_right = gather(floors[0], ceils[1], 'top_right')
    bottom_left = gather(ceils[0], floors[1], 'bottom_left')
    bottom_right = gather(ceils[0], ceils[1], 'bottom_right')

    interp_top = alphas[1] * (top_right - top_left) + top_left
    interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
    interp = alphas[0] * (interp_bottom - interp_top) + interp_top

    return interp

class SpecAugmentor(object):
    """3 way spectrogram augmentation :
        1) Time warping
        2) Frequency masking
        3) Time masking
    """

    def _spec_augment(mel_spectrogram, time_warping_para=80, frequency_masking_para=27,
                     time_masking_para=100, frequency_mask_num=1, time_mask_num=1, time_warp=False):
        """Spectrogram augmentation function from https://arxiv.org/abs/1904.08779
        3 steps processing :
        i) Time warping : depending on parameter time_warping_para
        ii) Frequency masking : depending on parameter frequency_masking_para and frequency_mask_num
        iii) Time masking : depending on parameter time_masking_para and time_mask_num

        Parameters
        ----------
        mel_spectrogram :       `SlidingWindowFeature` or (n_samples, n_features ) `numpy.ndarray`
            Features.
        time_warping_para:      time warp parameter W
        frequency_masking_para: frequency mask parameter F
        time_masking_para:      time mask parameter T
        frequency_mask_num:     number of frequency masks m_F
        time_mask_num:          number of time masks m_T

        Returns
        -------
        warped_masked_melspec : `SlidingWindowFeature` or (n_samples, n_features ) `numpy.ndarray`
            augmented spectrogram
        """
        v = mel_spectrogram.shape[0]
        tau = mel_spectrogram.shape[1]

        if time_masking_para > 0.2*tau:
            time_masking_para = int(0.2*tau)
        augmented_mel_spectrogram = mel_spectrogram.copy()

        # 1) Time warping
        if time_warp:
            augmented_mel_spectrogram = time_warper(torch.tensor(mel_spectrogram[np.newaxis, :]).float(), W=5)
            augmented_mel_spectrogram = augmented_mel_spectrogram.numpy().squeeze()

        # 2) Frequency masking
        for i in range(frequency_mask_num):
            f = np.random.uniform(low=0.0, high=frequency_masking_para)
            f = int(f)
            f0 = random.randint(0, v - f)
            augmented_mel_spectrogram[f0:f0 + f, :] = 0

        # 3) Time masking
        for i in range(time_mask_num):
            t = np.random.uniform(low=0.0, high=time_masking_para)
            t = int(t)
            t0 = random.randint(0, tau - t)
            augmented_mel_spectrogram[:, t0:t0 + t] = 0

        return augmented_mel_spectrogram

    def __call__(self, features, time_warping_para, frequency_masking_para,
                 time_masking_para, nb_frequency_masks, nb_time_masks, time_warp,
                 sliding_window=None):
        """Apply Google specAugment

        Parameters
        ----------
        features : `SlidingWindowFeature` or (n_samples, n_features ) `numpy.ndarray`
            Features.
        sliding_window : `SlidingWindow`, optional
            Not used.

        Returns
        -------
        normalized : `SlidingWindowFeature` or (n_samples, n_features ) `numpy.ndarray`
            Standardized features
        """

        if isinstance(features, SlidingWindowFeature):
            spec = features.data
        else:
            spec = features

        spec = SpecAugmentor._spec_augment(mel_spectrogram=spec,
                                            time_warping_para=time_warping_para,
                                            frequency_masking_para=frequency_masking_para,
                                            time_masking_para=time_masking_para,
                                            frequency_mask_num=nb_frequency_masks,
                                            time_mask_num=nb_time_masks,
                                           time_warp=time_warp)

        if isinstance(features, SlidingWindowFeature):
            return SlidingWindowFeature(spec, features.sliding_window)
        else:
            return spec
