var currentRegion = 0;
var regions = null;
var ids = null;
var refresh = true;

var left = 'ArrowLeft';
var right = 'ArrowRight';
var startR = 'Shift';
var endR = 'Control';

var PRECISION = (prodigy.config.precision / 1000);
var BEEP = (prodigy.config.beep === ('true'|| "True")) ;
var SPECTRO = (prodigy.config.spectrogram === ('true'|| "True")) ;
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

function loadScript(url, callback){
    var head = document.head;
    var script = document.createElement('script');
    script.type = 'text/javascript';
    script.src = url;
    script.onreadystatechange = callback;
    script.onload = callback;
    head.appendChild(script);
}

var loadSpectrogram = function(){
    loadScript("https://unpkg.com/wavesurfer.js/dist/plugin/wavesurfer.spectrogram.js",createSpectro);
}

var createSpectro = function(){
    window.wavesurfer.addPlugin(WaveSurfer.spectrogram.create({
        wavesurfer: window.wavesurfer,
        container: window.wavesurfer.container,
        labels: true
    })).initPlugin('spectrogram');
}

if(SPECTRO){
  loadScript("https://unpkg.com/wavesurfer.js", loadSpectrogram);
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
      150
    );
  }
}

function reloadWave(){
  regions = window.wavesurfer.regions.list;
  ids = Object.values(regions);
  ids.sort(compare);
  if(ids.length > 0){
    currentRegion = 0;
    ids[0].update({'color' : "rgba(0, 255, 0, 0.2)"});
  }
}

function switchCurrent(newId){
  if(ids.length > 0){
    ids[currentRegion].update({'color' : "rgba(255, 215, 0, 0.2)"});
    currentRegion = newId;
    ids[newId].update({'color' : "rgba(0, 255, 0, 0.2)"});
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
            if(ids.length > 0) ids[currentRegion].update({'color' : "rgba(255, 215, 0, 0.2)"});
            reloadWave();
            if(refresh){
              switchCurrent(0);
            }else{
              switchCurrent(ids.indexOf(e));
            }
          }, 5);
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
      re = window.wavesurfer.addRegion({'start' : pos,'end' : fin,'color' : "rgba(255, 215, 0, 0.2)"});
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
