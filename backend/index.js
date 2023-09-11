const express = require('express')
const cors = require('cors')
const app = express()
const multer = require('multer')
app.use(express.json())
app.use(cors())

const storage = multer.diskStorage({
    destination: function (req, file, cb) {
      cb(null, './public/images')
    },
    filename: function (req, file, cb) {
      cb(null, file.originalname)
    }
})
const upload = multer( {storage})

app.post("/upload", upload.single('file'), ( req, res)=>{
    try{
        console.log(req.file)
        res.status(200).send({success:true})
    }
    catch(error){
        console.log(error)
        res.status(501).send({success:false})
    }
})

app.listen(3001,()=>{
    console.log("server is running")
})