// script.js
// Reads the form, validates it, sends it to the FastAPI backend and shows the result.

var fieldNames = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"];

var form = document.getElementById("crop-form");
var resultBox = document.getElementById("result");

// Show the error message under one input box
function showError(name, message) {
  var errorElement = document.getElementById("error-" + name);
  var inputElement = document.getElementById(name);
  errorElement.textContent = message;
  if (message === "") {
    inputElement.classList.remove("invalid");
  } else {
    inputElement.classList.add("invalid");
  }
}

// Validate a field as soon as the user leaves the box
fieldNames.forEach(function (name) {
  document.getElementById(name).addEventListener("blur", function () {
    showError(name, checkValue(name, this.value));
  });
});

// Turn a probability like 0.8123 into "81.2%"
function toPercent(p) {
  return (p * 100).toFixed(1) + "%";
}

function showResult(data) {
  var html = "";

  if (data.status === "unusual") {
    resultBox.className = "result unusual";
    html += "<h2>Unusual input</h2>";
    html += "<p>" + data.message + "</p>";
    if (data.features_outside_training_range.length > 0) {
      html += "<p><b>Values outside the training range:</b> " + data.features_outside_training_range.join(", ") + "</p>";
    } else {
      html += "<p>Each value is normal on its own, but this combination is rare in the training data.</p>";
    }
    html += "<p class='small'>Unusual score: " + data.unusual_score + " (limit " + data.unusual_limit + ")</p>";
  } else {
    if (data.status === "confident") {
      resultBox.className = "result confident";
    } else {
      resultBox.className = "result low";
    }
    html += "<h2>Recommended crop: " + data.recommended_crop + "</h2>";
    html += "<p>Confidence: <b>" + toPercent(data.confidence) + "</b></p>";
    html += "<p>" + data.message + "</p>";
    html += "<h3>Top 3 crops</h3><ol>";
    data.top3.forEach(function (item) {
      html += "<li>" + item.crop + " - " + toPercent(item.probability) + "</li>";
    });
    html += "</ol>";
    html += "<p class='small'>Unusual score: " + data.unusual_score + " (limit " + data.unusual_limit + ")</p>";
  }

  resultBox.innerHTML = html;
}

form.addEventListener("submit", function (event) {
  event.preventDefault();   // stop the page from reloading

  // 1. check every field first
  var allValid = true;
  var values = {};
  fieldNames.forEach(function (name) {
    var text = document.getElementById(name).value;
    var message = checkValue(name, text);
    showError(name, message);
    if (message !== "") {
      allValid = false;
    } else {
      values[name] = Number(text.trim());
    }
  });

  if (!allValid) {
    resultBox.className = "result invalid-box";
    resultBox.textContent = "Invalid input: please fix the fields marked in red.";
    return;   // do not call the model
  }

  // 2. send to the backend
  resultBox.className = "result";
  resultBox.textContent = "Checking...";

  fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(values)
  })
    .then(function (response) {
      return response.json().then(function (data) {
        return { ok: response.ok, data: data };
      });
    })
    .then(function (answer) {
      if (!answer.ok) {
        // the backend also checks the input; show its messages
        var errors = answer.data.errors || {};
        Object.keys(errors).forEach(function (name) {
          showError(name, errors[name]);
        });
        resultBox.className = "result invalid-box";
        resultBox.textContent = "Invalid input: please fix the fields marked in red.";
      } else {
        showResult(answer.data);
      }
    })
    .catch(function () {
      resultBox.className = "result invalid-box";
      resultBox.textContent = "Could not reach the server. Is uvicorn running?";
    });
});

// Clear button
document.getElementById("clear-btn").addEventListener("click", function () {
  fieldNames.forEach(function (name) {
    document.getElementById(name).value = "";
    showError(name, "");
  });
  resultBox.className = "result";
  resultBox.textContent = "";
});
