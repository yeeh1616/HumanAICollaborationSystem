function save_summary(pid) {
    var summary = document.getElementById("w3review").value;
    var parmas = pid + "------" + summary;

    const xhttp = new XMLHttpRequest();
    xhttp.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
             var btn_save = document.getElementById("save_summary");
             var btn_next = document.getElementById("next");
             btn_save.disabled = true;
             btn_next.disabled = false;
        }
    };
    xhttp.open("POST", "/policies/save_summary");
    xhttp.send(parmas);
}

function summary_change() {
    var btn_save = document.getElementById("save_summary");
    btn_save.disabled = false;
}

function reload_summary(pid) {
    var btn = document.getElementById("btn_reload");

    if(btn.innerHTML=="Reload Summary"){
        const xhttp = new XMLHttpRequest();
        xhttp.onreadystatechange = function() {
            if (this.readyState == 4 && this.status == 200) {
                document.getElementById("w3review").value = xhttp.responseText;
            }
        };
        xhttp.open("POST", "/policies/reload_summary");
        xhttp.send(pid);
        btn.innerHTML="Recover Summary";
        document.getElementById("save_summary").disabled = false;
    }else {
        document.getElementById("w3review").value = "{{ policy.description }}";
        btn.innerHTML="Reload Summary";
        document.getElementById("save_summary").disabled = true;
    }
}