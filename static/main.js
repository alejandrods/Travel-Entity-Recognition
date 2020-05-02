
function onKeyDown(e) {
    if (e.keyCode === 32) { // tab key
        getNextWordFromService()
    }

}

function getNextWordFromService() {

    // get model type
    var input = document.getElementById("sentence").value.trim();
    console.log(input)

    var data = JSON.stringify({"input": input});
    var xhr = new XMLHttpRequest();
    xhr.withCredentials = true;

    console.log("DATA")
    console.log(data)

    xhr.addEventListener("readystatechange", function () {
    if (this.readyState === 4) {
      var data = JSON.parse(this.responseText)
//      document.getElementById("sentence").value = input + data.next_word;
      document.getElementById("pred").innerHTML = data.pred;

    }
    });

    // make prediction
    xhr.open("POST", "http://localhost:7000/predict");
    xhr.setRequestHeader("Content-Type", "application/json");

    xhr.send(data);
}

document.getElementById("sentence").addEventListener("keydown",onKeyDown);
