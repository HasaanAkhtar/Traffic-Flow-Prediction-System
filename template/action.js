
function myFunction() {
   
    function readTextFile(file, callback) {
        
        var oscat = document.getElementById("scat_origin").innerHTML;
			var dscat = document.getElementById("scat_dest").innerHTML;
			
			function getcoordinates(){
				let [lat1, long1] = oscat.split(",");
				var source = L.marker([lat1, long1]).addTo(mymap);
				source.setLatLng([lat1,long1]);

				let [lat2, long2] = dscat.split(",");
				var destination = L.marker([lat2, long2]).addTo(mymap);
				destination.setLatLng([lat2, long2]);
			}
        var rawFile = new XMLHttpRequest();
        rawFile.overrideMimeType("application/json");
        rawFile.open("GET", file, true);
        rawFile.onreadystatechange = function() {
            if (rawFile.readyState === 4 && rawFile.status == "200") {
                callback(rawFile.responseText);
            }
        }
        rawFile.send(null);
    }
    
    //usage:
    readTextFile("SCATs Accurate co-ordinates.json", function(text){
        var data = JSON.parse(text);
        console.log(data);
    });

}



function init() {
    myFunction;
     var form = document.getElementById("settings");//get refto the HTMLelement
     form.onsubmit = myFunction;//register the event listener
  }  
  window.addEventListener("load",init);