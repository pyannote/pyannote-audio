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

var numberAnnotations = 2;
var colorList = [];
var hex2rgba = (hex, alpha = 0.2) => {
  const [r, g, b] = hex.match(/\w\w/g).map(x => parseInt(x, 16));
  return `rgba(${r},${g},${b},${alpha})`;
};


/**
* Makes sure that the document is loaded before executing waitForElement()
* @see waitForElement()
*/
if(document.readyState !== 'loading') {
    waitForElement();
} else {
    document.addEventListener('DOMContentLoaded', function () {
        waitForElement();
    });
}

/**
* Remove label on region (created by prodigy)
* @param(r) region object
* @see waitForElement()
*/
function removeLabel(r){
    s = r.start;
    e = r.end;
    c = r.color;
    r.remove();
    window.wavesurfer.addRegion({"start" : s, "end" :e, 'color' : c, "resize": false, "drag": false});
}

/**
* Add reference and hypothesis regions on the corresponding wavesurfer instance
*/
function loadRegions(){
    for(var i=0; i < numberAnnotations;i++){
        (i == 0)? (regions = window.prodigy.content.reference) : (regions = window.prodigy.content.hypothesis)
        for (region in regions){
            var label = regions[region]['label'];
            if (!(label in colorList)){
              colorList.push(label);
            }
            var id = colorList.indexOf(label) % prodigy.config.custom_theme.palettes.audio.length;
            var color = prodigy.config.custom_theme.palettes.audio[id];
            color = hex2rgba(color);
            var re = window['wavesurfer'+i].addRegion({'start' : regions[region]['start'],'end' : regions[region]['end'],'color' : color, 'resize' : false, 'drag' : false, "attributes": {"label":regions[region]['label']}});
            addRegionLabel(re,label);
        }
    }
}

/**
* Create audio url for reference and hypothesis wavesurfer instance
* @see loadWave()
* @see LoadTrack()
*/
async function createURL(){
      var blob;
      var track = document.querySelector('#track');
      var src = track.children[0].src
      var blob = await (await fetch(src)).blob()
      var objectURL = URL.createObjectURL(blob);
      return objectURL;
}

/**
* Create the same wavesurfer object as on Prodigy interface for reference and hypothesis
* Those objects are muted
*/
async function loadWave(){
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
    for(var i=0; i < numberAnnotations;i++){
        window['wavesurfer'+i] = WaveSurfer.create(wdict);
        window['wavesurfer'+i].load(objectURL);
        window['wavesurfer'+i].setMute(true);
    }
    var l = document.querySelector('wave').appendChild(document.createElement("span"));
    l.textContent= "Errors";
    l.className = "title-wave";
    const nodeList = document.querySelectorAll('wave ~ wave');
    for(var i = 0; i < nodeList.length;i++){
      nodeList[i].style.marginTop = "30px";
      nodeList[i].style.backgroundColor = "#0000000a";
      l = nodeList[i].appendChild(document.createElement("span"));
      (i == 0)? (l.textContent = "Reference") : (l.textContent = "Hypothesis")
      l.className = "title-wave";
    }
}

/**
* Handle wavesurfer
* Tracking audio playback for reference and hypothesis wavesurfer
*/
async function waitForElement(){
    if((typeof window.wavesurfer !== "undefined") && (document.querySelector('#track') !== null)){
        window.wavesurfer.on('region-created', function(e){
         setTimeout(function(){
            if("label" in e){
              removeLabel(e);
            }},5);
        })
        window.wavesurfer.on('audioprocess', function(e){
          var time = e / window.wavesurfer.getDuration();
          for(var i=0; i < numberAnnotations;i++){
              window['wavesurfer'+i].seekAndCenter(time);
          }
        });
        window.wavesurfer.on('seek', function(e){
          for(var i=0; i < numberAnnotations;i++){
              window['wavesurfer'+i].seekTo(e);
          }
        });
        window.wavesurfer.on('zoom', function(e){
          for(var i=0; i < numberAnnotations;i++){
              window['wavesurfer'+i].zoom(e);
          }
        });
        window.wavesurfer.on('finish', function(e){
          for(var i=0; i < numberAnnotations;i++){
              window['wavesurfer'+i].seekTo(1);
          }
        });
        window.wavesurfer.on('pause', function(e){
          for(var i=0; i < numberAnnotations;i++){
              window['wavesurfer'+i].pause();
          }
        });
        await loadWave();
        loadRegions();
    }else{
       setTimeout(waitForElement, 250);
    }
}

/**
* Reload audio (when there is a new task)
* @see prodigyanswer
*/
async function loadTrack(){
  if(document.querySelector('#track') !== null){
    var objectURL = await createURL();
    for(var i=0; i < numberAnnotations;i++){
      window['wavesurfer'+i].load(objectURL);
      window['wavesurfer'+i].clearRegions();
    }
    loadRegions();
  }else{
    setTimeout(loadTrack, 250);
  }
}

// Check if it's a new prodigy task
document.addEventListener('prodigyanswer', async() => {
  loadTrack();
});

/**
* Add CSS label to reference and hypothesis region
* @see loadRegions()
* @param(r) region object
* @param(l) label
*/
function addRegionLabel(r,label){
   var s = r.element
   var l = s.appendChild(document.createElement("span"))
   l.textContent = label,
   l.className = "pyannote-region",
   l.style.color = "rgb(0, 0, 0)",
   l.style.background = r.color,
   r.label = label;
}
