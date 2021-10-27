var currentRegion = 0;
var regions = null;
var ids = null;

const left = 'ArrowLeft';
const right = 'ArrowRight';
const ctrl = 'Control';
const shift = 'Shift';

const PRECISION = (prodigy.config.precision / 1000);
const EXCERPT = 1;

var keysMap = {};

const audioCtx = new(window.AudioContext || window.webkitAudioContext)();

if( document.readyState !== 'loading' ) {
    waitForElement();
} else {
    document.addEventListener('DOMContentLoaded', function () {
        waitForElement();
    });
}

function beep() {
  var oscillator = audioCtx.createOscillator();
  var gainNode = audioCtx.createGain();

  oscillator.connect(gainNode);
  gainNode.connect(audioCtx.destination);

  gainNode.gain.value = 0.06;
  oscillator.frequency.value = 440;
  oscillator.type = "square";

  oscillator.start();

  setTimeout(
    function() {
      oscillator.stop();
    },
    150
  );
};

function reloadWave(){
  regions = window.wavesurfer.regions.list;
  ids = Object.keys(regions);
  if(ids.length > 0){
    currentRegion = 0;
    regions[ids[0]].update({'color' : "rgba(0, 255, 0, 0.2)"});
  }
}

function switchCurrent(newId){
  regions[ids[currentRegion]].update({'color' : "rgba(255, 215, 0, 0.2)"});
  currentRegion = ids.indexOf(newId);
  regions[newId].update({'color' : "rgba(0, 255, 0, 0.2)"});
}

function waitForElement(){
    if(typeof window.wavesurfer !== "undefined"){
        reloadWave();
        window.wavesurfer.on('region-created',function(e){
          if(currentRegion != 0){
            regions[ids[currentRegion]].update({'color' : "rgba(255, 215, 0, 0.2)"});
          }
          setTimeout(reloadWave, 10);
        });
        window.wavesurfer.on('region-click',function(e){
          switchCurrent(e.id);
        });
        window.wavesurfer.on('region-out',function(e){
          beep();
        });
        window.wavesurfer.on('region-removed',function(){
          reloadWave();
        });
    }else{
       setTimeout(waitForElement, 250);
    }
}

document.querySelector('#root').onkeydown = document.querySelector('#root').onkeyup = function(e){
    e = e || event;
    keysMap[e.key] = e.type == 'keydown';
    var pos = window.wavesurfer.getCurrentTime();
    var audioEnd = window.wavesurfer.getDuration();
    var region = regions[ids[currentRegion]];

    if(keysMap[left] && !keysMap[right]){
      if(keysMap[ctrl] && !keysMap[shift]){
        if((region.start - PRECISION) <= 0){
          region.update({'start' : 0});
          window.wavesurfer.play(0, region.end);
        }else{
          region.update({'start' : region.start - PRECISION });
          window.wavesurfer.play(region.start, region.end);
        }
      }else if(keysMap[shift] && !keysMap[ctrl]){
        if((region.end - PRECISION) > region.start){
          region.update({'end' : region.end - PRECISION });
          window.wavesurfer.play(region.end - EXCERPT, region.end);
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
      if(keysMap[ctrl] && !keysMap[shift]){
        if(region.start + PRECISION < region.end){
          region.update({'start' : region.start + PRECISION });
          window.wavesurfer.play(region.start, region.end);
        }
      }else if(keysMap[shift] && !keysMap[ctrl]){
        if((region.end + PRECISION) >= audioEnd){
          region.update({'end' : audioEnd });
          window.wavesurfer.play(region.end - EXCERPT, region.end);
        }else{
          region.update({'end' : region.end + PRECISION });
          window.wavesurfer.play(region.end - EXCERPT, region.end);
        }
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
    }else if (keysMap['n']){
      var fin = pos + 1;
      if(fin > audioEnd) fin = audioEnd;
      re = window.wavesurfer.addRegion({'start' : pos,'end' : fin,'color' : "rgba(255, 215, 0, 0.2)"});
      window.wavesurfer.fireEvent('region-update-end',re);
      setTimeout(switchCurrent, 11, re.id);
    }else if(keysMap['Backspace']){
      regions[ids[currentRegion]].remove();
    }else if(keysMap['ArrowUp']){
      if(currentRegion == (ids.length - 1)){
        switchCurrent(ids[0]);
      }else{
        switchCurrent(ids[currentRegion + 1]);
      }
    }else if(keysMap['u']){
      reloadWave();
    }
}
