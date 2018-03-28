// -*- coding: utf-8 -*-

var model = null;
var runner = null;
var sheet = null;
var idx = 0;
var SIZE = 64;

function callback_progressbar(current, total) {
    var pct = Math.round(current / total * 100);
    p.p = pct;
    p.update();
}

// refer http://cya.sakura.ne.jp/java/loading.htm
function display_inference() {
    var inf = document.querySelector('#calc');
    var myY = (window.innerHeight !== undefined) ? window.innerHeight : document.body.clientHeight;
    var myX = (window.innerWidth !== undefined) ? window.innerWidth : document.body.clientWidth;
    inf.style.top = (myY/2-70 < 0) ? 0 : myY/2-70;
    inf.style.left = (myX/2-300 <0) ? 0 : myX/2-300;
    inf.style.visibility = "visible";
    return inf;
}

function hide_inference() {
    var inf = document.querySelector('#calc');
    inf.style.visibility = "hidden";
}

// ref: https://www.nishishi.com/javascript-tips/setinterval-passage.html

var sec = 0;
function showPassage() {
    sec ++;
    var msg = sec + "sec";
    document.getElementById("sec").innerText = msg;
}

function startTimer() {
    sec = 0;
    PassageId = setInterval('showPassage()', 1000);
    document.getElementById("inference").style.disabled = true;
    document.getElementById("sec").innerText = "";
};

function stopTimer() {
    clearInterval(PassageId);
    document.getElementById("inference").style.disabled = false;
};

// ref: https://codepen.io/utano320/pen/XXpeav
var canv;
var ctx;
var cw = 400;
var ch = 400;
var mouseX;
var mouseY;

function init_canv() {
    canv = document.getElementById('axisCanvas');
    canv.width = cw;
    canv.height = cw;
    ctx = canv.getContext('2d');
    canv.onclick = function(e) {
	ctx.clearRect(0, 0, cw, ch);
	var rect = e.target.getBoundingClientRect();
	mouseX = e.clientX - Math.floor(rect.left) - 2;
	mouseY = e.clientY - Math.floor(rect.top) - 2;
	
	ctx.beginPath();
	ctx.arc(mouseX, mouseY, 5, 0, Math.PI * 2, false);
	ctx.fill();

	var z1 = (mouseX - cw/2) / (cw /2) * 2.5;
	var z2 = (mouseY - ch/2) / (ch /2) * 2.5;

	document.getElementById("val_z1").innerHTML = z1;
	document.getElementById("val_z2").innerHTML = z2;

	start();
    };
};

window.onload = function() {
    init_canv();
    model = WebDNN.load("../webdnn", {
	progressCallback: callback_progressbar,
	backendOrder: ["webassembly"]}).then(function (r) {
    	var stat = document.querySelector("#status");
    	stat.textContent = "Loaded.";
	runner = r;
	var v = document.getElementById("prog-bar");
	v.style.visiblility = "hidden";
    });
};

function start() {
    console.log("start");
    var z1 = document.querySelector("#val_z1");
    var z2 = document.querySelector("#val_z2");
    var vec = new Float32Array(2);
    vec[0] = parseFloat(z1.innerText);
    vec[1] = parseFloat(z2.innerText);
    console.log("runner called");
    display_inference();
    startTimer();
    var x = runner.getInputViews()[0];
    var y = runner.getOutputViews()[0];
    x.set(vec);
    runner.run().then(function () {
	console.log("output");
	var ret = y.toActual();
	var cvs = document.querySelector("#output");
	var clip = [];
	// var newimg = new Float32Array(ret.length);
	// for (var i = 0; i < ret.length; i ++) {
	//     newimg[i] = ret[i] * -127.5 + 127.5;
	// }
	WebDNN.Image.setImageArrayToCanvas(ret, 64, 64, cvs, {
	    // scale: [0.5, 0.5, 0.5],
	    // bias: [0.5, 0.5, 0.5],
	    scale: [127.5, 127.5, 127.5],
	    // scale: [-127.5, -127.5, -127.5],
	    bias: [127.5, 127.5, 127.5],
	    color: WebDNN.Image.Color.RGB,
	    order: WebDNN.Image.Order.CHW
	});
	stopTimer();
	hide_inference();
    });
}
