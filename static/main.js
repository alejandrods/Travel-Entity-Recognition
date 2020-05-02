// Send request when spacebar
function onKeyDown(e) {
    if (e.keyCode === 32) {
        getNextWordFromService()
    }

}

// Function to send request to model
function getNextWordFromService() {
    var input = document.getElementById("sentence").value.trim();
    console.log(input)

    // Build data json
    var data = JSON.stringify({"input": input});
    var xhr = new XMLHttpRequest();
    xhr.withCredentials = true;

    xhr.addEventListener("readystatechange", function () {
    if (this.readyState === 4) {
      var data = JSON.parse(this.responseText)
      document.getElementById("pred").innerHTML = data.pred;
        }
    });

    // Request prediction
    xhr.open("POST", "http://localhost:7000/predict");
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.send(data);
}

document.getElementById("sentence").addEventListener("keydown",onKeyDown);
