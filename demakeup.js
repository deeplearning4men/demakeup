var STRIDE = 32;
var TILE_SZ = 64;
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



function get_tile(u, v, data)
{
    var flat = new Float32Array(3 * TILE_SZ * TILE_SZ);
    var p = 0;
    for (var i = 0 ; i < TILE_SZ ; i++)
    {
        for (var j = 0 ; j < TILE_SZ ; j++)
        {
            var y = u + i;
            var x = v + j;
            var start = 4 * (y*SZ + x);
            flat[p++] = data[start + 0] / 255;
            flat[p++] = data[start + 1] / 255;
            flat[p++] = data[start + 2] / 255;
        }
    }
    return flat;
}

// counter: overlap counter
// data: data we want to make
// tile: tile input to accumulate
// u, v: defines where in data the tile data is accumulated.
function accumulate_output(counter, data, tile, u, v)
{
    var p = 0;
    for (var i = 0 ; i < TILE_SZ ; i++)
    {
        for (var j = 0 ; j < TILE_SZ ; j++)
        {
            var y = u + i;
            var x = v + j;
            var start = 4 * (y*SZ + x);
            data[start] += tile[p++];
            data[start + 1] += tile[p++];
            data[start + 2] += tile[p++];
            data[start + 3] = 255; // alpha channel

            counter[start + 0] += 1;
            counter[start + 1] += 1;
            counter[start + 2] += 1;
            counter[start + 3] = 1; // alpha channel
        }
    }
}



function renderOutput(outputs)
{
    // The input is in RGB, not RGBA.
    // Need to draw manually.
    var ctx = document.getElementById('output-canvas').getContext('2d');
    var imageData = ctx.createImageData(SZ, SZ);

    // overlapping region counter
    var counter = new Float32Array(4 * SZ * SZ);
    var data = new Float32Array(4 * SZ * SZ); // rgba
    for (var i = 0 ; i < counter.length ; i++)
    {
        counter[i] = 0;
        data[i] = 0;
    }

    // accumulate each tile to data. do counting too.
    // ... to average 'em later.
    var tile_index = 0;
    for (var i = 0 ; i <= SZ - TILE_SZ ; i += STRIDE)
    {
        for (var j = 0 ; j <= SZ - TILE_SZ ; j += STRIDE)
        {
            var tile = outputs[tile_index];
            accumulate_output(counter, data, tile, i, j);
            tile_index++;
        }
    }

    // Time to average for good looking merge.
    for (var i = 0 ; i < data.length ; i++)
    {
        data[i] /= counter[i];
    }

    for (var i = 0 ; i < data.length ; i++)
    {
        imageData.data[i] = 255*data[i];
    }
    ctx.putImageData(imageData, 0, 0);

    // These work, why not putImageData?
    //ctx.fillStyle="red"
    //ctx.fillRect(10,10,50,50)
}



function resizeImages()
{
    // SZ x SZ is not too good to see.
    //document.getElementById('output-canvas').style.width = "256px";
    //document.getElementById('input-canvas').style.width = "256px";
    // Nah, no resize.
}



function split_tiles(imageData)
{
    var data = imageData.data;
    var flat_tiles = new Array();

    for (var i = 0 ; i <= SZ - TILE_SZ ; i += STRIDE)
    {
        for (var j = 0 ; j <= SZ - TILE_SZ ; j += STRIDE)
        {
            var tile = get_tile(i, j, data);
            flat_tiles.push(tile);
        }
    }

    return flat_tiles;
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



function runModel(ctx)
{
    var imageData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
    var spinner = showSpin();

    var flat_tiles = split_tiles(imageData);

    var gpu = document.getElementById('gpu');

    var model = new KerasJS.Model({
      filepaths: {
        model: 'model.json',
        weights: 'model_weights.buf',
        metadata: 'model_metadata.json'
      },
      gpu: gpu.checked
    });

    model.ready().then(() => {
        var jobid = 0;
        var outputs = new Array(flat_tiles.length);

        function next()
        {
            if (jobid < flat_tiles.length)
            {
                var inputData = {
                    'input_1': flat_tiles[jobid]
                };

                model.predict(inputData).then(outputData => {
                    // Draw onto output canvas.
                    spinner.stop();
                    var data = outputData['convolution2d_11'];
                    outputs[jobid] = data;
                    jobid++;
                    next(); // do some weird recursion to run the job in sequence.
                });

            }
            else
            {
                renderOutput(outputs);
                //resizeImages();
            }
        }

        next();
    });
}
