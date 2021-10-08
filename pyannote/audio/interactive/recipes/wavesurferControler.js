var currentRegion = 0;
var regions = null;
var ids = null;
const keys = prodigy.config.keys
const EL = keys['EL'];
const ER = keys['ER'];
const SR = keys['SR'];
const SL = keys['SL'];
const N  = keys['N'];
const R  = keys['R'];
const PRECISION = 0.1;
const EXCERPT = 0.2;

if( document.readyState !== 'loading' ) {
    waitForElement();
} else {
    document.addEventListener('DOMContentLoaded', function () {
        waitForElement();
    });
}

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
          setTimeout(reloadWave, 100);
        });
        window.wavesurfer.on('region-click',function(e){
          switchCurrent(e.id);
        });
        window.wavesurfer.on('region-update-end',function(e){
          switchCurrent(e.id);
        });
        window.wavesurfer.on('region-removed',function(){
          reloadWave();
        });
    }else{
       setTimeout(waitForElement, 250);
    }
}


document.querySelector('#root').addEventListener('keypress', function (event) {
    var region = regions[ids[currentRegion]];
    oldStart = region.start;
    oldEnd = region.end;
    if(event.keyCode === EL){
      if((oldEnd - PRECISION) > oldStart){
        region.update({'end' : oldEnd - PRECISION });
        window.wavesurfer.play(region.end - EXCERPT, region.end);
      }
    } else if(event.keyCode === ER){
      var audioEnd = window.wavesurfer.getDuration();
      if((oldEnd + PRECISION) >= audioEnd){
        region.update({'end' : audioEnd });
        window.wavesurfer.play(region.end - EXCERPT, region.end);
      }else{
        region.update({'end' : oldEnd + PRECISION });
        window.wavesurfer.play(region.end - EXCERPT, region.end);
      }
    } else if(event.keyCode === SL){
      if((oldStart - PRECISION) <= 0){
        region.update({'start' : 0});
        window.wavesurfer.play(0, region.start + EXCERPT);
      }else{
        region.update({'start' : oldStart - PRECISION });
        window.wavesurfer.play(region.start, region.start + EXCERPT);
      }
    } else if(event.keyCode === SR){
      if(oldStart + PRECISION < oldEnd){
        region.update({'start' : oldStart + PRECISION });
        if((oldStart + PRECISION - EXCERPT) < 0){
          window.wavesurfer.play(0, region.start + EXCERPT);
        }else{
          window.wavesurfer.play(region.start, region.start + EXCERPT);
        }
      }
    }
});


document.querySelector('#root').addEventListener('keydown', function (event) {
    if(event.keyCode === N){
        if(currentRegion == (ids.length - 1)){
          switchCurrent(ids[0]);
        }else{
          switchCurrent(ids[currentRegion + 1]);
        }
    }else if (event.keyCode === R){
        regions[ids[currentRegion]].remove();
    }else if (event.keyCode === 81){
        reloadWave();
    }
});
