var handle = ''
var progress = 'Free'
var download_link
var re_loop

function setCookie(cname, cvalue, exdays) {
    var d = new Date();
    d.setTime(d.getTime() + (exdays * 24 * 60 * 60 * 1000));
    var expires = "expires=" + d.toUTCString();
    document.cookie = cname + "=" + cvalue + "; " + expires;
}

function getCookie(cname) {
    var name = cname + "=";
    var ca = document.cookie.split(';');
    for (var i = 0; i < ca.length; i++) {
        var c = ca[i];
        while (c.charAt(0) == ' ') c = c.substring(1);
        if (c.indexOf(name) != -1)
            return c.substring(name.length, c.length);
    }
    return "";
}

function clearCookie(name) {
    setCookie(name, "", -1);
}


function Init() {
    var w = document.getElementsByClassName("div-main")
    for (var i = 0; i < w.length; i++) {
        w[i].style.display = "none"
    }
    document.getElementById("div-upload").style.display = ""
    pb = document.getElementById('pb')
    pb.innerText = '0%'
    pb.style.width = '0%'
    progress = 'Free'
    handle = getCookie('TASKHANDLE')
    if (handle) {
        Working()
    }
}

function ReInit() {
    document.getElementById("alert").style.display = "none"
    var w = document.getElementsByClassName("div-main")
    for (var i = 0; i < w.length; i++) {
        w[i].style.display = "none"
    }
    document.getElementById("div-upload").style.display = ""
    pb = document.getElementById('pb')
    pb.innerText = '0%'
    pb.style.width = '0%'
    progress = 'Free'
    ClearTask()
}

function CancelTask() {
    ReInit()
}

function Working() {
    var w = document.getElementsByClassName("div-main")
    for (var i = 0; i < w.length; i++) {
        w[i].style.display = "none"
    }
    document.getElementById("div-process").style.display = ""
    var download = '#'
    re_loop = setInterval(function () {
        ReStatus()
        if (progress == 'Done') {
            clearInterval(re_loop)
            var w = document.getElementsByClassName("div-main")
            for (var i = 0; i < w.length; i++) {
                w[i].style.display = "none"
            }
            document.getElementById("div-download").style.display = ""
            document.getElementById("bnt-download").href = download_link
        }
    }, 1000)
}

function ReStatus() {
    var alert = document.getElementById("alert")
    $.ajax({
        url: '/status',
        type: "get",
        success: function (res) {
            if (res) {
                if (res['Status'] < 100) {

                    progress = res['Process']
                    var p = Math.floor(res['FinishImg'] / res['Count'] * 100)
                    pb = document.getElementById('pb')
                    pb.innerText = p + '%'
                    pb.style.width = p + '%'
                    download_link = res['Download']
                } else if (res['Status'] = 703) {
                    alert.innerHTML =
                        'Task can not be found. <a href="#" class="alert-link" onclick="ReInit()">Click to return.</a>'
                    alert.style.display = ""
                }
            }
        }
    })
}

function UploadPic() {
    var alert = document.getElementById("alert")
    var form = document.getElementById('upload-pic'),
        formData = new FormData(form);

    $.ajax({
        url: "/upload",
        type: "post",
        data: formData,
        processData: false,
        contentType: false,
        success: function (res) {
            if (res) {
                if (res['Status'] < 100) {
                    var w = document.getElementsByClassName("div-main")
                    for (var i = 0; i < w.length; i++) {
                        w[i].style.display = "none"
                    }
                    document.getElementById("div-process").style.display = ""
                    Working()
                    alert.style.display = "none"
                } else {
                    alert.innerText = res['Status'] + ' ' + res['StatusText']
                    alert.style.display = ""
                }
            } else {
                alert.innerText = 'No return value!'
                alert.style.display = ""
            }
            $("#file-btn").val("");
        },
        timeout: 5000,
        error: function (err) {
            $("#file-btn").val("");
            alert.innerText = err.status + ' ' + err.statusText
            alert.style.display = ""
        }
    })
}

function ClearTask() {
    clearInterval(re_loop)
    clearCookie('TASKHANDLE')
}