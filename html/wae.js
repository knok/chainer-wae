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

window.onload = function() {
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
