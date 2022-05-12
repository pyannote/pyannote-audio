var snd = new Audio();
var idTimer = 0;

if(document.readyState !== 'loading') {
    if(typeof window.wavesurfer !== "undefined"){
        setTimeout(handleWavesurfer,25);
    }else{
        handleWavesurfer();
    }
}else{
    document.addEventListener('DOMContentLoaded', function () {
        handleWavesurfer();
    });
}

function checkRegion(label){
    for(var idr in window.wavesurfer.regions.list){
        var region = window.wavesurfer.regions.list[idr];
        if(label == region.label){
            return true;
        }
    }
    return false;
}

function changePlaceholderColor(label, color){
    var labels = document.querySelectorAll("label[for="+label+"]");
    if(labels.length > 0){
        if(color.includes("rgb")){
          color = color.split('0.2)');
          color = color[0];
          color = color+'1)';
        }
        labels[0].style.color = color
    }
}


function changeDisplayPlaceholder(label, hide){
    var labels = document.querySelectorAll("label[for="+label+"]");
    if(labels.length > 0){
        labels[0].parentElement.hidden = hide;
    }
}

function displayPlaceholder(){
    for(var idr in window.wavesurfer.regions.list){
        var region = window.wavesurfer.regions.list[idr];
        changeDisplayPlaceholder(region.label, false);
    }
}

function clearSpan(){
    var spans = document.querySelectorAll("span[class='c01140']");
    for(var span in spans){
        if(typeof spans[span].innerHTML !== "undefined"){
            if(spans[span].innerHTML.includes('SPEAKER_')){
                var spk = spans[span].innerHTML.split('SPEAKER_');
                spans[span].innerHTML = spk[1];
            }
        }
    }
}

function handleWavesurfer(){
    if(typeof window.wavesurfer !== "undefined"){
        addSounds();
        window.wavesurfer.on('region-created', function(e){
            setTimeout(function (){
                changeDisplayPlaceholder(e.label, false);
                //TODO : Overkill, update only the new one
                clearSpan();
            }, 5);
        });
        window.wavesurfer.on('region-update-end', function(e){
            setTimeout(clearSpan, 5);
            setTimeout(changeDisplayPlaceholder, 5, e.label, false);
        });
        window.wavesurfer.on('region-removed',function(e){
            if(!checkRegion(e.label)){
                changeDisplayPlaceholder(e.label, true);
            }
        });
    }else{
        setTimeout(handleWavesurfer, 250);
    }
}

function addSounds(){
    var labels = document.querySelectorAll('label.prodigy-label');
    if(labels.length > 0){
      var sounds = window.prodigy.content.sounds;
      clearSpan();
      for (var label of labels) {
          var i = (window.prodigy.content.config.labels.indexOf(label.dataset.prodigyLabel) % window.prodigy.config.custom_theme.palettes.audio.length);
          var color = window.prodigy.config.custom_theme.palettes.audio[i];
          label.style.color = color;
          changePlaceholderColor(label.dataset.prodigyLabel, color);
          changeDisplayPlaceholder(label.dataset.prodigyLabel, true);

          label.onmouseover = (e) => {
              var val = e.srcElement.dataset.prodigyLabel;
              if(val in sounds){
                  snd.pause();
                  clearInterval(idTimer);
                  snd = new Audio(sounds[val]);
                  snd.play();
                  idTimer = setInterval(function () {
                      ele = e.srcElement;
                      i = (window.prodigy.content.config.labels.indexOf(val) % window.prodigy.config.custom_theme.palettes.audio.length);
                      color = window.prodigy.config.custom_theme.palettes.audio[i];
                      ele.style.backgroundColor = (ele.style.backgroundColor == '' ? color : '');
                  }, 200);
              }
          }
          label.onmouseleave = (e) => {
              snd.pause();
              clearInterval(idTimer);
              e.srcElement.style.backgroundColor = '';
          }
      }
      displayPlaceholder();
    }else{
        setTimeout(addSounds,250);
    }
}

document.addEventListener('prodigyanswer', e => {
  addSounds();
});
