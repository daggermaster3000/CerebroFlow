function testDashboard(){
    
    var imagePath = document.getElementById("image_path").files
    console.log(imagePath)
    eel.test()

}
var paths;
var output_path

async function getPaths() {
    paths = await eel.select_input()();
    console.log(paths)
    let file_div = document.getElementById('file-div');
    file_div.innerHTML = "Selected: "+paths;
    }

async function getOutputFolder() {
    output_path = await eel.select_output()();
    let folder_div = document.getElementById('folder-div');
    folder_div.innerHTML = "Selected: "+output_path;
    }
