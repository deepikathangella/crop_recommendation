// validation.js
// Checks one input value typed by the user.
// It returns an error message, or an empty string "" if the value is fine.

// allowed ranges (same as the backend)
var limits = {
  N: [0, 300],
  P: [0, 200],
  K: [0, 400],
  temperature: [-10, 60],
  humidity: [0, 100],
  ph: [0, 14],
  rainfall: [0, 3000]
};

function checkValue(name, text) {
  // remove spaces at the start and end
  text = String(text).trim();

  // 1. empty?
  if (text === "") {
    return "Invalid input: please enter a value";
  }

  // 2. must look like a number: digits, optional minus sign, optional decimal part.
  // This rejects words like "hello", symbols like "@#$", and things like "12abc" or "1e5".
  var numberPattern = /^-?\d+(\.\d+)?$/;
  if (!numberPattern.test(text)) {
    return "Invalid input: enter numbers only";
  }

  // 3. inside the allowed range?
  var number = Number(text);
  var low = limits[name][0];
  var high = limits[name][1];
  if (number < low || number > high) {
    return "Invalid input: must be between " + low + " and " + high;
  }

  return "";   // no error
}

// This lets the same file be tested with Node (it does nothing in the browser)
if (typeof module !== "undefined") {
  module.exports = { checkValue: checkValue, limits: limits };
}
