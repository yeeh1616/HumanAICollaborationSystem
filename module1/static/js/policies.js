function search(){
    var pid = document.getElementById("pidSearch").value;
    var url = '';

    if(pid == ""){
        url = '/policies/all';
    }else {
        url = '/policies/' + pid + '/search';
    }

    location.replace(url);
}