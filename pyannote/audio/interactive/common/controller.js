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


var currentRegion = 0;
var regions = null;
var ids = null;
var refresh = true;

var left = 'ArrowLeft';
var right = 'ArrowRight';
var startR = 'Shift';
var endR = 'Control';

var PRECISION = (prodigy.config.precision / 1000);
var BEEP = prodigy.config.beep;
var EXCERPT = 1;

var keysMap = {};

var audioCtx = new(window.AudioContext || window.webkitAudioContext)();

if(document.readyState !== 'loading') {
    if(typeof window.wavesurfer !== "undefined"){
        setTimeout(waitForElement,25);
    }else{
        waitForElement();
    }
} else {
    document.addEventListener('DOMContentLoaded', function () {
        waitForElement();
    });
}

function compare(region1, region2){
  if(region1.start < region2.start){
    return -1;
  }else if (region1.start > region2.start){
    return 1;
  }else{
    return 0;
  }
}

function beep() {
  if(BEEP){
    var oscillator = audioCtx.createOscillator();
    var gainNode = audioCtx.createGain();

    oscillator.connect(gainNode);
    gainNode.connect(audioCtx.destination);

    gainNode.gain.value = 0.1;
    oscillator.frequency.value = 440;
    oscillator.type = "square";

    oscillator.start();

    setTimeout(
      function() {
        oscillator.stop();
      },
      150  // FIXME: should depend on the value of "precision"
    );
  }
}

function activeRegion(e){
  e.element.style.borderTop = "3px solid";
  e.element.style.borderBottom = "3px solid";
}

function deactiveRegion(e){
  e.element.style.borderTop = "";
  e.element.style.borderBottom = "";
}

function reloadWave(){
  regions = window.wavesurfer.regions.list;
  ids = Object.values(regions);
  ids.sort(compare);
  if(ids.length > 0){
    currentRegion = 0;
    activeRegion(ids[0]);
  }
}

function switchCurrent(newId){
  if(ids.length > 0){
    deactiveRegion(ids[currentRegion]);
    currentRegion = newId;
    activeRegion(ids[newId])
    if(refresh){
       window.wavesurfer.seekTo(0);
    }else{
      var time = (ids[currentRegion].start) / (window.wavesurfer.getDuration());
      window.wavesurfer.seekTo(time);
    }
  }
}

function waitForElement(){
    if(typeof window.wavesurfer !== "undefined"){
        reloadWave();
        window.wavesurfer.on('region-created', function(e){
          setTimeout(function(){
            if(ids.length > 0) deactiveRegion(ids[currentRegion]);
            reloadWave();
            if(refresh){
              switchCurrent(0);
            }else{
              switchCurrent(ids.indexOf(e));
            }
          }, 5);
        });
        window.wavesurfer.on('region-dblclick',function(e){
          re = window.wavesurfer.addRegion({'start' : e.start,'end' : e.end});
          window.wavesurfer.fireEvent('region-update-end',re);
          e.remove();
        });
        window.wavesurfer.on('region-click',function(e){
          switchCurrent(ids.indexOf(e));
        });
        window.wavesurfer.on('region-out',function(e){
          beep();
        });
        window.wavesurfer.on('region-removed',function(){
          if(currentRegion == (ids.length - 1)){
            var newId = 0;
          }else{
            var newId = currentRegion;
          }
          reloadWave();
          if(ids.length > 0) switchCurrent(newId);
        });
    }else{
       setTimeout(waitForElement, 250);
    }
}

document.addEventListener('prodigyanswer', e => {
  refresh = true;
})

document.querySelector('#root').onkeydown = document.querySelector('#root').onkeyup = function(e){
    e = e || event;
    keysMap[e.key] = e.type == 'keydown';
    var pos = window.wavesurfer.getCurrentTime();
    var audioEnd = window.wavesurfer.getDuration();
    var region = ids[currentRegion];
    refresh = false;

    if(keysMap[left] && !keysMap[right]){
      if(keysMap[startR] && !keysMap[endR]){
        if((region.start - PRECISION) <= 0){
          region.update({'start' : 0});
          window.wavesurfer.play(0, region.end);
       }else{
          region.update({'start' : region.start - PRECISION });
          window.wavesurfer.play(region.start, region.end);
        }
      }else if(keysMap[endR] && !keysMap[startR]){
        var startTime = region.end - EXCERPT;
        if(startTime < region.start) startTime = region.start;
        if((region.end - PRECISION) > region.start){
          region.update({'end' : region.end - PRECISION });
          window.wavesurfer.play(startTime, region.end);
        }
      }else{
        if(keysMap['w']){
          var time = (pos - PRECISION*2) / audioEnd;
        }else{
          var time = (pos - PRECISION) / audioEnd;
        }
        if(time < 0) time = 0;
        window.wavesurfer.pause();
        window.wavesurfer.seekTo(time);
      }
    }else if(keysMap[right] && !keysMap[left]){
      if(keysMap[startR] && !keysMap[endR]){
        if(region.start + PRECISION < region.end){
          region.update({'start' : region.start + PRECISION });
          window.wavesurfer.play(region.start, region.end);
        }
      }else if(keysMap[endR] && !keysMap[startR]){
        if(!window.wavesurfer.isPlaying()){
          var startTime = region.end - EXCERPT;
          if(startTime < region.start) startTime = region.start;
        }else{
          var startTime = pos;
        }
        if((region.end + PRECISION) >= audioEnd){
           region.update({'end' : audioEnd });
        }else{
          region.update({'end' : region.end + PRECISION });
        }
        window.wavesurfer.play(startTime, region.end);
      }else{
        if(keysMap['w']){
          var time = (pos + PRECISION*2) / audioEnd;
        }else{
          var time = (pos + PRECISION) / audioEnd;
        }
        if(time > 1) time = 1;
        window.wavesurfer.pause();
        window.wavesurfer.seekTo(time);
      }
    }else if (keysMap['ArrowUp'] && keysMap['Shift']){
      var fin = pos + 1;
      if(fin > audioEnd) fin = audioEnd;
      re = window.wavesurfer.addRegion({'start' : pos,'end' : fin});
      window.wavesurfer.fireEvent('region-update-end',re);
    }else if(keysMap['Backspace'] || (keysMap['ArrowDown'] && keysMap['Shift'])){
      ids[currentRegion].remove();
    }else if(keysMap['ArrowUp']){
      if(currentRegion == (ids.length - 1)){
        switchCurrent(0);
      }else{
        switchCurrent(currentRegion + 1);
      }
    }else if(keysMap['ArrowDown']){
      if(currentRegion == 0){
        switchCurrent(ids.length - 1);
      }else{
        switchCurrent(currentRegion - 1);
      }
    }else if(keysMap['u']){
      reloadWave();
    }
}
