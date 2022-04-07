// The MIT License (MIT)
//
// Copyright (c) 2021 CNRS
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

var numberAnnotations = prodigy.config.number_annotations;

var colorList = [];
var hex2rgba = (hex, alpha = 0.2) => {
    const [r, g, b] = hex.match(/\w\w/g).map(x => parseInt(x, 16));
    return `rgba(${r},${g},${b},${alpha})`;
};

/**
* Makes sure that the document is loaded before executing wait()
*/
if (document.readyState !== 'loading') {
    wait();
} else {
    document.addEventListener('DOMContentLoaded', function () {
        wait();
    });
}

/**
* Add regions for all annotations on their corresponding wavesurfer instance
*/
function loadRegions() {
    for (var i = 0; i < numberAnnotations; i++) {
        var regions = window.prodigy.content.annotations[i];
        for (region in regions) {
            var color;
            var label = regions[region]['label'];
            if (!colorList.includes(label)) {
                colorList.push(label);
            }
            var id = colorList.indexOf(label) % prodigy.config.custom_theme.palettes.audio.length;
            var color = prodigy.config.custom_theme.palettes.audio[id];
            color = hex2rgba(color);
            var re = window['wavesurfer' + i].addRegion({ 'start': regions[region]['start'], 'end': regions[region]['end'], 'color': color, 'resize': false, 'drag': false, "attributes": { "label": label } });
            addRegionLabel(re, label);
        }
    }
}

/**
* Create audio url
* @see loadWave()
* @see prodigyanswer
*/
async function createURL() {
    var blob;
    var track = document.querySelector('#track');
    var src = track.children[0].src
    var blob = await (await fetch(src)).blob()
    var objectURL = URL.createObjectURL(blob);

    return objectURL;
}

/**
* Create the same wavesurfer object as on Prodigy interface for all annotations
* Those objects are muted
*/
async function loadWave() {
    var objectURL = await createURL();
    var wdict = {
        container: window.wavesurfer.container,
        audioRate: 1,
        autoCenter: true,
        autoCenterImmediately: false,
        backend: "WebAudio",
        barGap: 2,
        barHeight: 1,
        barMinHeight: null,
        barRadius: 2,
        barWidth: 0,
        cursorColor: "#333",
        cursorWidth: 1,
        fillParent: true,
        forceDecode: false,
        height: 128,
        hideScrollbar: false,
        interact: false,
        loopSelection: true,
        maxCanvasWidth: 4000,
        mediaControls: false,
        mediaType: "audio",
        normalize: false,
        partialRender: false,
        pixelRatio: 2,
        progressColor: "#583fcf",
        waveColor: "violet",
        removeMediaElementOnDestroy: true,
        responsive: false,
        scrollParent: true,
        skipLength: 2,
        splitChannels: false,
        plugins: [
            WaveSurfer.regions.create({})
        ]
    };
    for (var i = 0; i < numberAnnotations; i++) {
        window['wavesurfer' + i] = WaveSurfer.create(wdict);
        window['wavesurfer' + i].load(objectURL);
        window['wavesurfer' + i].setMute(true);
        window['wavesurfer' + i].on('region-click', function (e) {
            var label = e.attributes.label;
            clickOnLabel(label);
            var re = window.wavesurfer.addRegion({ 'start': e.start, 'end': e.end, 'color': e.color });
            window.wavesurfer.fireEvent('region-update-end', re);
        });
    }
    var l = document.querySelector('wave').appendChild(document.createElement("span"));
    l.textContent = "Output";
    l.className = "title-wave";
    const nodeList = document.querySelectorAll('wave ~ wave');
    for (var i = 0; i < nodeList.length; i++) {
        nodeList[i].style.marginTop = "30px";
        nodeList[i].style.backgroundColor = "#0000000a";
        l = nodeList[i].appendChild(document.createElement("span"));
        l.textContent = "Input " + (i + 1);
        l.className = "title-wave";
        l.onclick = function () { test(i) };
        l.setAttribute("onclick", "addAllRegions(" + i + ");");
    }
}

/**
* Handle click on annotation label
* Add all regions of the clicked annotation on the main wavesurfer instance (the prodigy's one)
* @param(i) indice of annotation (wavesurfer instance)
*/
function addAllRegions(i) {
    var rs = window['wavesurfer' + i].regions.list;
    num = Object.values(rs);
    for (var i in num) {
        var region = num[i];
        var label = region.attributes.label;
        clickOnLabel(label);
        var re = window.wavesurfer.addRegion({ 'start': region.start, 'end': region.end, 'color': region.color });
        window.wavesurfer.fireEvent('region-update-end', re);
    }
}

/**
* Handle wavesurfer
* Tracking audio playback for all annotations
*/
async function wait() {
    if (document.querySelector('#track') !== null) {
        await loadWave();
        loadRegions();
        window.wavesurfer.on('audioprocess', function (e) {
            var time = e / window.wavesurfer.getDuration();
            for (var i = 0; i < numberAnnotations; i++) {
                window['wavesurfer' + i].seekAndCenter(time);
            }
        });
        window.wavesurfer.on('seek', function (e) {
            for (var i = 0; i < numberAnnotations; i++) {
                window['wavesurfer' + i].seekTo(e);
            }
        });
        window.wavesurfer.on('zoom', function (e) {
            for (var i = 0; i < numberAnnotations; i++) {
                window['wavesurfer' + i].zoom(e);
            }
        });
        window.wavesurfer.on('finish', function (e) {
            for (var i = 0; i < numberAnnotations; i++) {
                window['wavesurfer' + i].seekTo(1);
            }
        });
        window.wavesurfer.on('pause', function (e) {
            for (var i = 0; i < numberAnnotations; i++) {
                window['wavesurfer' + i].pause();
            }
        });
    } else {
        setTimeout(wait, 250);
    }
}

/**
* Load new audio and clear regions for all annotations wavesurfer when there is a new task
*/
document.addEventListener('prodigyanswer', async () => {
    var objectURL = await createURL();
    for (var i = 0; i < numberAnnotations; i++) {
        window['wavesurfer' + i].load(objectURL);
        window['wavesurfer' + i].clearRegions();
    }
    loadRegions();
});

/**
* Add CSS label to reference and hypothesis region
* @see loadRegions()
* @param(r) region object
* @param(l) label
*/
function addRegionLabel(r, label) {
    var s = r.element
    var l = s.appendChild(document.createElement("span"))
    l.textContent = label,
    l.className = "pyannote-region",
    l.style.color = "rgb(0, 0, 0)",
    l.style.background = r.color,
    r.label = label;
}
