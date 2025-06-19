document.getElementById('input-file').addEventListener('change', function(event) {
  var reader = new FileReader();
  reader.onload = function () {
    var output = document.getElementById('output-image');
    output.src = reader.result;
  }
  reader.readAsDataURL(event.target.files[0]);
});
