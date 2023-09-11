import {  Box, Button, Card, HStack, Icon, Image, Input, Stack, Text } from '@chakra-ui/react'
import React, { useState } from 'react'
import {MdAddAPhoto} from 'react-icons/md'

import axios from 'axios'

const Search = () => {
  const [input, setinput] = useState("")
  const [file,setfile] = useState()

  const handlefile = (e)=>{
  
   setfile(e.target.files[0])

  }
  const handlesubmit = ()=>{
    console.log("hiihi")
    const formdata = new FormData()
    formdata.append('file',file)
    console.log(file)
    axios.post("http://localhost:3001/upload",formdata)
    .then(res => console.log(res.data.success)).catch(err => console.log(err))

  }
  return (
    <>
        <HStack justify={"space-between"} spacing={"2px"}>
            <Input onChange={(e)=>setinput(e.target.value)} color={"red"} />
            <Input type='file'  onChange={handlefile} />
        </HStack>
        <Button onClick={handlesubmit} mt="6px"> Submit</Button>
        {/* <Image src = {file}/> */}
        <Stack>
          <Text> {input}</Text>
          
          
        </Stack>
        
    </>
  )
}

export default Search