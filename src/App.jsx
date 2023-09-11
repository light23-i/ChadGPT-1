import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import { Box } from '@chakra-ui/react'
import Search from './components/Search'

function App() {
  const [count, setCount] = useState(0)

  return (
    
      <Search/>
    
  )
}

export default App