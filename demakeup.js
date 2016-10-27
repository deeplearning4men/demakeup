var SZ = 128;

function renderImage(file)
{
    var reader = new FileReader();
    reader.onload = function(event){
        var the_url = event.target.result;
        //of course using a template library like handlebars.js is a better
        //solution than just inserting a string
        var img = new Image();
        img.onload = function() {
            var canvas = document.getElementById('input-canvas');
            //canvas.style.width = "64px";
            //canvas.style.height = "64px";
            var ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0, SZ, SZ);
            runModel(ctx);
        };
        img.src = the_url;
        // Image needs some time to draw too.
        // That's why I'm using onload for img too.
    };
    //when the file is read it triggers the onload event above.
    reader.readAsDataURL(file);
}


function renderOutput( outputData )
{
    var data = outputData[ 'convolution2d_11' ];

    // The input is in RGB, not RGBA.
    // Need to draw manually.
    var ctx = document.getElementById('output-canvas').getContext('2d');
    var imageData = ctx.createImageData(SZ, SZ);
    var j = 0;
    for (var i = 0 ; i < 3*SZ*SZ ; i += 3)
    {
        imageData.data[ j++ ] = 255 * data[ i+0 ];
        imageData.data[ j++ ] = 255 * data[ i+1 ];
        imageData.data[ j++ ] = 255 * data[ i+2 ];
        imageData.data[ j++ ] = 255;
    }
    ctx.putImageData(imageData, 0, 0);

    // These work, why not putImageData?
    //ctx.fillStyle="red"
    //ctx.fillRect(10,10,50,50)
}



function flatten( imageData )
{
	var data = imageData.data;

    var flat = new Float32Array(3 * SZ * SZ);
    var j = 0;
    for(var i=0 ; i < data.length ; i += 4)
    {
        flat[j++] = data[i + 0] / 255;
        flat[j++] = data[i + 1] / 255;
        flat[j++] = data[i + 2] / 255;
        // And, don't copy alpha channel.
    }

    return flat;
}



function showSpin()
{
    var opts = {
      lines: 13, // The number of lines to draw
      length: 28, // The length of each line
      width: 14, // The line thickness
      radius: 50, // The radius of the inner circle
      scale: 0.5, // Scales overall size of the spinner
      corners: 1, // Corner roundness (0..1)
      color: '#000', // #rgb or #rrggbb or array of colors
      opacity: 0.25, // Opacity of the lines
      rotate: 0, // The rotation offset
      direction: 1, // 1: clockwise, -1: counterclockwise
      speed: 1, // Rounds per second
      trail: 60, // Afterglow percentage
      fps: 20, // Frames per second when using setTimeout() as a fallback for CSS
      zIndex: 2e9, // The z-index (defaults to 2000000000)
      className: 'spinner', // The CSS class to assign to the spinner
      top: '50%', // Top position relative to parent
      left: '50%', // Left position relative to parent
      shadow: false, // Whether to render a shadow
      hwaccel: false, // Whether to use hardware acceleration
      position: 'absolute' // Element positioning
    };
    var target = document.getElementById('preview');
    var spinner = new Spinner(opts).spin(target);
    return spinner;
}



function runModel( ctx )
{
	var imageData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
    var spinner = showSpin();

    var flat = flatten(imageData);

	var gpu = document.getElementById('gpu');

    var model = new KerasJS.Model({
      filepaths: {
        model: 'model.json',
        weights: 'model_weights.buf',
        metadata: 'model_metadata.json'
      },
      gpu: gpu.checked
    });

    model.ready().then( () => {
        var inputData = {
            'input_1': flat
        };

        model.predict(inputData).then(outputData => {
            // Draw onto output canvas.
            spinner.stop();
            renderOutput(outputData);
        });
    });
}
