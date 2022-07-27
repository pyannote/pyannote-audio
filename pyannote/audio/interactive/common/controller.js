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

// Can't create constant because prodigy reload this file every batch
//

// Index of the active region in the array "ids"
var currentRegion = 0;

// Array of all regions (window.wavesurfer.regions.list)
var regions = null;

// Array of all regions ids
var ids = null;

/*
* Boolean allowing to know if the interface has just been refreshed.
* When a new chunk arrives, wavesurfer triggers the region-created events but
* its event listener must not do exactly the same as when a user triggered it.
* It must leave the first region as an active region (and not the last one created)
* and the cursor must not be set to the beginning of the region but to the beginning
* of the audio file, at 0s.
*
* Refresh is put to false whenever it can be assumed that an action has been taken by a user.
*/
var refresh = true;

// Keyboard configuration
var left = 'ArrowLeft';
var right = 'ArrowRight';
var startR = prodigy.config.left || 'Shift';
var endR = prodigy.config.right || 'Control';

// Cursor precision
var PRECISION = (prodigy.config.precision / 1000);

// Beep when the player reaches the end of a region.
var BEEP = prodigy.config.beep;

// How many seconds before the cursor will be played when the end of a region is changed
var EXCERPT = 1;

/*
* Since the computer only passes one keystroke at a time, an array is created to keep track
* of multiple keys. The array can then be used to check for one or more keys at once.
*/
var keysMap = {};

// Beta version for logging user interaction...
var logsInfo = [];

/**
* Handle web audio for beep
* @see beep()
*/
var audioCtx = new(window.AudioContext || window.webkitAudioContext)();

/**
* Makes sure that the document is loaded before executing waitForElement()
* Add a small timeout if wavesurfer is already defined (useful when a new batch is coming),
* the time for prodigy to update wavesurfer.
* @see waitForElement()
*/
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

/**
* Compare if a region is before or after another one
* Useful to sort global variable 'ids'
* @see reloadWave()
* @param {region1} Region object
* @param {region2} Region Object
* @return {number} -1 if region1 start before, 1 otherwise, 0 if the identical
*/
function compare(region1, region2){
  if(region1.start < region2.start){
    return -1;
  }else if (region1.start > region2.start){
    return 1;
  }else{
    return 0;
  }
}

/**
* Simulate a click on prodigy's label radio button
* Only useful for "review recipe"
* Note: might break with future versions of Prodigy
* @param {label} Label to click on
*/
function clickOnLabel(label){
  document.querySelector("input[type=radio][value=\'"+label+"\']").click()
}

/**
* Update prodigy.content with all regions from window.wavesurfer.regions.list (thus, all regions that can be seen in the interface)
* Discuss in this issue :
* https://support.prodi.gy/t/weird-interaction-between-window-prodigy-update-and-wavesurfer/5450
*/
function updateContent(){
  var regions = window.wavesurfer.regions.list;
  var content = [];
  for (var id in regions){
    var region = regions[id];
    content.push({start : region.start, end : region.end, label : region.label, id : region.id, color : region.color});
  }
  window.prodigy.update({audio_spans : content});
}

/**
* Create a beep sound from scratch
* You can adjust the gain, frequency (here its A 440) and duration
*/
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

/**
* Change CSS style for the selected region
* @param {e} Region object
*/
function activeRegion(e){
  e.element.style.borderTop = "3px solid";
  e.element.style.borderBottom = "3px solid";
}

/**
* @see activeRegion()
* Undo CSS change
* @param {e} Region object
*/
function deactiveRegion(e){
  e.element.style.borderTop = "";
  e.element.style.borderBottom = "";
}

/**
* Update global variables 'regions' and 'ids' with regions in window.wavesurfer.regions.list
* Put the first one as "active" and update the variable currentRegion
*/
function reloadWave(){
  regions = window.wavesurfer.regions.list;
  ids = Object.values(regions);
  ids.sort(compare);
  if(ids.length > 0){
    currentRegion = 0;
    activeRegion(ids[0]);
  }
}

/**
* Switch selected region
* Update var currentRegion
* Place wavesurfer cursor at the beginning of the new region or the beginning of the file if it's a new prodigy task
* @see activeRegion() / deactiveRegion()
* @param {ids} Ids of the region to be selected
* @param {seek} Boolean, true if we want the cursor to be at the start of the new active region
*               False otherwise (region-click, region-removed)
*/
function switchCurrent(newId, seek=true){
  if(ids.length > 0){
    deactiveRegion(ids[currentRegion]);
    currentRegion = newId;
    activeRegion(ids[newId])
    if(refresh){
       window.wavesurfer.seekTo(0);
    }else{
      var time = (ids[currentRegion].start) / (window.wavesurfer.getDuration());
      if(seek) window.wavesurfer.seekTo(time);
    }
  }
}

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


/**
* Handle wavesurfer regions
* Add event listener to some wavesurfer event
*/
function waitForElement(){
    if(typeof window.wavesurfer !== "undefined"){
        reloadWave();
        // Change Color of placeholders
        var labels = document.querySelectorAll('label.prodigy-label');
        if(labels.length > 0){
          for (var label of labels) {
            var i = (window.prodigy.content.config.labels.indexOf(label.dataset.prodigyLabel) % window.prodigy.config.custom_theme.palettes.audio.length);
            var color = window.prodigy.config.custom_theme.palettes.audio[i];
            label.style.color = color;
            var name = label.dataset.prodigyLabel;
            changePlaceholderColor(name, color);
          }
        }
        // Select created region or the first one if it's a new task
        window.wavesurfer.on('region-created', function(e){
          if(!refresh){
              logsInfo.push({"action" : "create", "timestamp"  : Date.now(), "type" : "mouse"});
              window.prodigy.update({logs : logsInfo});
          }
          setTimeout(function(){
            if(ids.length > 0) deactiveRegion(ids[currentRegion]);
            reloadWave();
            if(refresh){
              switchCurrent(0);
            }else{
              switchCurrent(ids.indexOf(e));
              window.wavesurfer.play(e.start);
            }
          }, 5);
        });
        // Change region label (by remove the old one and create a new one with proper label)
        window.wavesurfer.on('region-dblclick',function(e){
          logsInfo.push({"action" : "label change", "timestamp"  : Date.now()});
          refresh = false;
          re = window.wavesurfer.addRegion({'start' : e.start,'end' : e.end});
          e.remove();
          window.wavesurfer.fireEvent('region-update-end',re);
          // Remove "create", "delete" and "mouse change" action created by addregion, remove and region-update-end
          logsInfo.pop();
          logsInfo.pop();
          logsInfo.pop();
        });
        // Select region on click
        window.wavesurfer.on('region-click',function(e){
          refresh = false;
          switchCurrent(ids.indexOf(e), false);
        });
        // Beep when region end
        window.wavesurfer.on('region-out',function(e){
          beep();
        });
        // @see updateContent()
        window.wavesurfer.on('region-update-end', function(e){
          logsInfo.push({"action" : "mouse change", "timestamp"  : Date.now()});
          window.prodigy.update({logs : logsInfo});
          updateContent();
        });
        // @see updateContent()
        // Switch selected region when deleted
        window.wavesurfer.on('region-removed',function(e){
          logsInfo.push({"action" : "delete", "timestamp"  : Date.now(), "type" : "mouse"});
          window.prodigy.update({logs : logsInfo});
          updateContent();
          if(currentRegion == (ids.length - 1)){
            var newId = 0;
          }else{
            var newId = currentRegion;
          }
          reloadWave();
          if(ids.length > 0) switchCurrent(newId, false);
        });
    }else{
       setTimeout(waitForElement, 250);
    }
}

// Check if it's a new prodigy task
document.addEventListener('prodigyanswer', e => {
  logsInfo = [];
  refresh = true;
})

/**
*  Keyboard controller
* | Key 1             | Key 2              | Command                                          |
* | -------------     | -------------      | ------------                                     |
* | Arrows left/right | [W]                | Move Cursor [speed up]                           |
* | Shift             | Arrows left/right  | Change start of current segment                  |
* | Control           | Arrows left/right  | Change end of current segment                    |
* | Arrows up/down    |                    | Change current segment to the next/precedent one |
* | Shift             | Arrows up/[down]   | Create [or remove] segment                       |
* | Backspace         |                    | Remove current segment                           |
*/
document.querySelector('#root').onkeydown = document.querySelector('#root').onkeyup = function(e){
    e = e || event;
    keysMap[e.key] = e.type == 'keydown';
    var pos = window.wavesurfer.getCurrentTime();
    var audioEnd = window.wavesurfer.getDuration();
    var region = ids[currentRegion];
    refresh = false;

    // If the focus is on a text field
    if (document.activeElement.type == 'text'){
      // Active main element if Enter is pressed
      if(keysMap['Enter']){
        var main = document.querySelector('main');
        main.focus();
      }else{
        return;
      }
    }

    // If Left is pressed
    if(keysMap[left] && !keysMap[right]){
      // If Shift is pressed
      if(keysMap[startR] && !keysMap[endR]){
        // Shortens start if possible
        logsInfo.push({"action" : "left change", "timestamp"  : Date.now()});
        if((region.start - PRECISION) <= 0){
          region.update({'start' : 0});
          window.wavesurfer.fireEvent('region-update-end',region);
          window.wavesurfer.play(0, region.end);
        }else{
          region.update({'start' : region.start - PRECISION });
          window.wavesurfer.fireEvent('region-update-end',region);
          window.wavesurfer.play(region.start, region.end);
        }
        // Remove "mouse change" log
        logsInfo.pop();
        window.prodigy.update({logs : logsInfo});
      // If Ctrl is pressed
      }else if(keysMap[endR] && !keysMap[startR]){
        var startTime = region.end - EXCERPT;
        if(startTime < region.start) startTime = region.start;
        // Shortens end if possible
        logsInfo.push({"action" : "right change", "timestamp"  : Date.now()});
        if((region.end - PRECISION) > region.start){
          region.update({'end' : region.end - PRECISION });
          window.wavesurfer.fireEvent('region-update-end',region);
          window.wavesurfer.play(startTime, region.end);
          // Remove "mouse change" log
          logsInfo.pop();
        }
        window.prodigy.update({logs : logsInfo});
      }else{
        // Else change cursor position
        // Speed up naviguation if W is pressed
        if(keysMap['w']){
          var time = (pos - PRECISION*2) / audioEnd;
        }else{
          var time = (pos - PRECISION) / audioEnd;
        }
        if(time < 0) time = 0;
        window.wavesurfer.pause();
        window.wavesurfer.seekTo(time);
      }
    // If Right is pressed
    }else if(keysMap[right] && !keysMap[left]){
      // If Shift is pressed
      if(keysMap[startR] && !keysMap[endR]){
        // Extend start if possible
        logsInfo.push({"action" : "leftChange", "timestamp"  : Date.now()});
        if(region.start + PRECISION < region.end){
          region.update({'start' : region.start + PRECISION });
          window.wavesurfer.fireEvent('region-update-end',region);
          window.wavesurfer.play(region.start, region.end);
          // Remove "mouse change" log
          logsInfo.pop();
        }
        window.prodigy.update({logs : logsInfo});
      // If Ctrl is pressed
      }else if(keysMap[endR] && !keysMap[startR]){
        // Extend end if possible (while keep playing the audio)
        logsInfo.push({"action" : "right change", "timestamp"  : Date.now()});
        if(!window.wavesurfer.isPlaying()){
          var startTime = region.end - EXCERPT;
          if(startTime < region.start) startTime = region.start;
        }else{
          var startTime = pos;
        }
        if((region.end + PRECISION) >= audioEnd){
           region.update({'end' : audioEnd });
           window.wavesurfer.fireEvent('region-update-end',region);
           // Remove "mouse change" log
           logsInfo.pop();
        }else{
          region.update({'end' : region.end + PRECISION });
          window.wavesurfer.fireEvent('region-update-end',region);
          // Remove "mouse change" log
          logsInfo.pop();
        }
        window.wavesurfer.play(startTime, region.end);
        window.prodigy.update({logs : logsInfo});
      }else{
        // Else change cursor position
        // Speed up naviguation if W is pressed
        if(keysMap['w']){
          var time = (pos + PRECISION*2) / audioEnd;
        }else{
          var time = (pos + PRECISION) / audioEnd;
        }
        if(time > 1) time = 1;
        window.wavesurfer.pause();
        window.wavesurfer.seekTo(time);
      }
    // If Up and shift is pressed : new region
    }else if (keysMap['ArrowUp'] && keysMap['Shift']){
      var fin = pos + 1;
      if(fin > audioEnd) fin = audioEnd;
      re = window.wavesurfer.addRegion({'start' : pos,'end' : fin});
      window.wavesurfer.fireEvent('region-update-end',re);
      // Remove "mouse change" and "create - mouse" log
      logsInfo.pop();
      logsInfo.pop();
      logsInfo.push({"action" : "create", "timestamp"  : Date.now(), "type" : "keyboard"});
      window.prodigy.update({logs : logsInfo});
    // If Down and Shift or Backspace: delete region
    // Check backspace for diarization text field
    }else if(keysMap['Backspace'] || (keysMap['ArrowDown'] && keysMap['Shift'])){
      ids[currentRegion].remove();
      // Remove "delete - mouse" log
      logsInfo.pop();
      logsInfo.push({"action" : "delete", "timestamp"  : Date.now(), "type" : "keyboard"});
      window.prodigy.update({logs : logsInfo});
    // If Up/Down @see switchCurrent
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
