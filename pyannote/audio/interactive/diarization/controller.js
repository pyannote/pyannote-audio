var snd = new Audio();
var idTimer = 0;

/**
* Make sure that the document is loaded before executing handleWavesurfer
* @see handleWavesurfer()
*/
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

/**
* Check if there is at least one region with this label
* @see handleWavesurfer()
* @param {label} label to check
*/
function checkRegion(label){
    for(var idr in window.wavesurfer.regions.list){
        var region = window.wavesurfer.regions.list[idr];
        if(label == region.label){
            return true;
        }
    }
    return false;
}

//THIS FUNCTION IS NOW ON THE MAIN CONTROLLER JS
/**
* Change the color of textfield's title
* @see addSounds()
* @param {label}
* @param {color}
*/
/*
function changePlaceholderColor(label, color){
    var labels = document.querySelectorAll("label[for="+label+"]");
    if(labels.length > 0){
        if(color.includes("rgb")){
          color = color.split('0.2)');
          color = color[0];
          color = color+'1)';
        }
        labels[0].style.color = color;
    }
}
*/

/**
* Hide or display textfield
* @see displayPlaceholder()
* @see handleWavesurfer()
* @see addSounds()
* @param {label} - label of textfield
* @param {hide} - boolean
*/
function changeDisplayPlaceholder(label, hide){
    var labels = document.querySelectorAll("label[for="+label+"]");
    if(labels.length > 0){
        labels[0].parentElement.parentElement.style.display = (hide ? "none" : "")
    }
}

/**
* Display textfields for all wavesurfer's regions label
* @see addSounds()
*/
function displayPlaceholder(){
    for(var idr in window.wavesurfer.regions.list){
        var region = window.wavesurfer.regions.list[idr];
        changeDisplayPlaceholder(region.label, false);
    }
}


/**
* Create "sound" emoji next to the prodigy labels
* Start the corresponding audio excerpt when the mouse is over
* @see addSounds()
* @param {label}
*/
function createEmojiSound(label){
    var span = label.children[0];
    var cloneSpan = span.cloneNode(true);
    span.innerHTML = "\ud83d\udd0a";
    span.style.fontSize = "20px";
    label.appendChild(cloneSpan);

    span.onmouseover = (e) => {
        var sounds = window.prodigy.content.sounds;
        var val = e.srcElement.parentElement.dataset.prodigyLabel;
        snd.pause();
        clearInterval(idTimer);
        snd = new Audio(sounds[val]);
        snd.play();
    }
    span.onmouseleave = (e) => {
        snd.pause();
        clearInterval(idTimer);
        e.srcElement.style.visibility = "";
    }
}

/**
* Change the regions label from "SPEAKER_XX" to "XX"
* Support only on Chrome from now
* !! use class='c01140' check if prodigy change
*/
function clearSpan(){
    //TODO: Check span class in other browsers
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

/**
* Add event listener to some wavesurfer event
*/
function handleWavesurfer(){
    if(typeof window.wavesurfer !== "undefined"){
        addSounds();
        //Display textfield if region is created
        window.wavesurfer.on('region-created', function(e){
            setTimeout(function (){
                changeDisplayPlaceholder(e.label, false);
                //TODO : Overkill, update only the new one
                clearSpan();
            }, 5);
        });
        //Check textfield if region is rename
        window.wavesurfer.on('region-update-end', function(e){
            setTimeout(clearSpan, 5);
            setTimeout(changeDisplayPlaceholder, 5, e.label, false);
        });
        //Hide textfield if region is removed
        window.wavesurfer.on('region-removed',function(e){
            if(!checkRegion(e.label)){
                changeDisplayPlaceholder(e.label, true);
            }
        });
    }else{
        setTimeout(handleWavesurfer, 250);
    }
}

/**
* Add sounds to Prodigy's label (they are in window.prodigy.content.sounds)
* And the same time change their colors and the color of their corresponding textfield
* @see changePlaceholderColor()
* @see changeDisplayPlaceholder()
*/
function addSounds(){
    var labels = document.querySelectorAll('label.prodigy-label');
    if(labels.length > 0){
      var sounds = window.prodigy.content.sounds;
      clearSpan();
      for (var label of labels) {
          var i = (window.prodigy.content.config.labels.indexOf(label.dataset.prodigyLabel) % window.prodigy.config.custom_theme.palettes.audio.length);
          var color = window.prodigy.config.custom_theme.palettes.audio[i];
          label.style.color = color;
          var name = label.dataset.prodigyLabel;
          //changePlaceholderColor(name, color);
          changeDisplayPlaceholder(name, true);
          if (name in sounds) createEmojiSound(label);
      }
      displayPlaceholder();
    }else{
        setTimeout(addSounds,250);
    }
}

// We suppose that batch_size = 1
// Some change is needed otherwise
/*
document.addEventListener('prodigyanswer', e => {
  addSounds();
});
*/
