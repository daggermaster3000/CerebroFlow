function testDashboard(){
    
    var imagePath = document.getElementById("image_path").files
    console.log(imagePath)
    eel.test()

}
let paths
async function getPaths(){
    paths = eel.select_input();
    
}
