import React from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Dashboard from './pages/Dashboard'
import Forecast from './pages/Forecast'
import Recommendations from './pages/Recommendations'
import HistoricalTrends from './pages/HistoricalTrends'

function App() {
  return (
    <Router>
      <div className="App">
        <Navbar />
        <main>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/forecast" element={<Forecast />} />
            <Route path="/recommendations" element={<Recommendations />} />
            <Route path="/historical-trends" element={<HistoricalTrends />} />
          </Routes>
        </main>
      </div>
    </Router>
  )
}

export default App