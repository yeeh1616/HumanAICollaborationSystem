function text_change(btn_save_id){
    var btn_save = document.getElementById(btn_save_id);
    btn_save.disabled = false;
}

function highlighting(qid,answerStr) {
    let divs = document.getElementById("summary").children;
    let answers = answerStr.split("|");

    for(i=0; i<divs.length; i++){
        if(divs[i].nodeName == "DIV"){
            divs[i].innerHTML = divs[i].innerHTML.replaceAll("<span style=\"background-color: #FFFF00\">","");
            divs[i].innerHTML = divs[i].innerHTML.replaceAll("</span>","");

            let senOld = divs[i].innerHTML;
            let senNew = senOld;

            for(j=0; j<answers.length; j++){
                senNew = senNew.replaceAll(answers[j], "<span style=\"background-color: #FFFF00\">" + answers[j] + "</span>")
            }

            divs[i].innerHTML = senNew;
        }
    }
}

function highlighting_multichoice(policyId, questionId) {
    // 1. 找到policy文本的div对象
    var policyNode = document.getElementById("summary");

    var ele = document.getElementsByName(questionId+'_answer');
    var optionId = "";

    for(i=0; i<ele.length; i++){
        if(ele[i].checked){
            optionId = ele[i].id;
        }
    }

    const xhttp = new XMLHttpRequest();
    xhttp.onreadystatechange = function(){
        if(this.readyState == 4 && this.status == 200){
            // clean policy div
            // read the returned policy turn it from json into object
            //
            var json = xhttp.responseText;
            var obj = JSON.parse(json);
            policyNode.innerHTML = "";

            for (const [key, graphs] of Object.entries(obj)) {
                var div = document.createElement("div");
                for (const g of Object.entries(graphs)) {
                    var sen = g[1];
                    var sentence = sen["sentence"];
                    var score = parseFloat(sen["score"].substring(0,3));

                    // 3. 生成一个span对象，加在div对象
                    var span = document.createElement("span");
                    span.innerHTML = sentence;
                    // 4. 如果score >= 0.8
                    // 5. 如果socre >= 0.7
                    // 6. 如果socre >= 0.5
                    // 7. 如果socre < 0.5
                    if(score>=0.9){
                        span.style.backgroundColor = '#ff9f00';
                    }else if(score>=0.8 && score<0.9){
                        span.style.backgroundColor = '#c37400';
                    }else if(score>=0.7 && score <0.8){
                        span.style.backgroundColor = '#fff300';
                    }else if(score>=0.6 && score <0.7){
                        span.style.backgroundColor = '#fffba1';
                    }else if(score>=0.5 && score <0.6){
                        span.style.backgroundColor = '#008eff';
                    }else if(score>=0.4 && score <0.5){
                        span.style.backgroundColor = '#81c7ff';
                    }else {

                    }
                    div.appendChild(span);
                }
                policyNode.append(div);
                policyNode.append(document.createElement("br"));
            }
        }
    };
    xhttp.open("POST", "/policies/highlighting");
    xhttp.send(policyId + "------" + questionId + "------" + optionId);
}

function save1(btn, qid, pid, column) {
    var answerName = qid + "_answer";
    var options = document.getElementById(answerName);
    var label = document.getElementById(qid + "_label");

    for(i = 0; i < options.length; i++) {
        if(options[i].checked) {
            var parmas = '{"pid":"' + pid + '","qid":"' + qid + '","answer":"' + options[i].value + '","column":"' + column + '"}';

            const xhttp = new XMLHttpRequest();
            xhttp.onreadystatechange = function() {
                if (this.readyState == 4 && this.status == 200) {
                    btn.disabled=true;
                }
            };
            xhttp.open("POST", "/policies/save");
            xhttp.send(parmas);
            break;
        }
    }
}

function save2(btn, qid, pid, column) {
    var answerId = qid + "_answer";
    var label = document.getElementById(qid + "_label");
    var answer = document.getElementById(answerId).value;
    var parmas = '{"pid":"' + pid + '","qid":"' + qid + '","answer":"' + answer + '","column":"' + column + '"}';

    const xhttp = new XMLHttpRequest();
    xhttp.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
            btn.disabled=true;
        }
    };
    xhttp.open("POST", "/policies/save");
    xhttp.send(parmas);
}