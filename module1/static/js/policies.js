function search(){
    var pid = document.getElementById("pidSearch").value;
    var url = '';

    if(pid == ""){
        url = '/policies/searchAll';
    }else {
        url = '/policies/' + pid + '/search';
    }

    location.replace(url);
}

function handleChange(checkbox) {
    let x = document.cookie
  .split(';')
  .reduce((res, c) => {
    const [key, val] = c.trim().split('=').map(decodeURIComponent)
    const allNumbers = str => /^\d+$/.test(str);
    try {
      return Object.assign(res, { [key]: allNumbers(val) ?  val : JSON.parse(val) })
    } catch (e) {
      return Object.assign(res, { [key]: val })
    }
  }, {});
    let x2 = x['cb'];
    if(checkbox.checked == true){
        alert(''+checkbox.id+'checked');
    }else{
        alert(''+checkbox.id+'unchecked');
   }
}