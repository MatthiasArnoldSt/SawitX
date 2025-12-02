import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { Download } from 'lucide-react'

const HistoricalTrends = () => {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchHistoricalData()
  }, [])

  const fetchHistoricalData = async () => {
    try {
      setLoading(true)
      const response = await axios.get('/api/historical-trends')
      setData(response.data)
    } catch (error) {
      console.error('Error fetching historical data:', error)
    } finally {
      setLoading(false)
    }
  }

  const exportCSV = async () => {
    try {
      const response = await axios.get('/api/export-csv', {
        responseType: 'blob'
      })
      
      const url = window.URL.createObjectURL(new Blob([response.data]))
      const link = document.createElement('a')
      link.href = url
      link.setAttribute('download', `ffb_forecast_${new Date().toISOString().split('T')[0]}.csv`)
      document.body.appendChild(link)
      link.click()
      link.remove()
      window.URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Error exporting CSV:', error)
    }
  }

  if (loading) {
    return <div className="loading">Loading historical trends...</div>
  }

  // Combine past and forecast data for the chart
  const chartData = [
    ...(data.past_data || []).map(item => ({
      ...item,
      type: 'Historical'
    })),
    ...(data.forecast || []).map(item => ({
      ...item,
      type: 'Forecast'
    }))
  ]

  return (
    <div className="container">
      <div style={styles.header}>
        <div>
          <h1>Historical Trends</h1>
          <p>Compare historical performance with future forecasts</p>
        </div>
        <button onClick={exportCSV} className="btn btn-primary" style={styles.exportBtn}>
          <Download size={16} />
          Export CSV
        </button>
      </div>

      <div className="card">
        <div style={styles.chartContainer}>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="date" 
                tick={{ fontSize: 12 }}
                interval="preserveStartEnd"
              />
              <YAxis 
                tick={{ fontSize: 12 }}
                tickFormatter={(value) => `RM${value.toFixed(2)}`}
              />
              <Tooltip 
                formatter={(value) => [`RM ${value.toFixed(2)}`, 'Price']}
                labelFormatter={(label) => `Date: ${label}`}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="price" 
                stroke="#3b82f6" 
                strokeWidth={2}
                dot={false}
                name="FFB Price"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div style={styles.stats}>
          <div style={styles.stat}>
            <span style={styles.statLabel}>Historical Data Points</span>
            <strong style={styles.statValue}>
              {data.past_data?.length || 0}
            </strong>
          </div>
          <div style={styles.stat}>
            <span style={styles.statLabel}>Forecast Period</span>
            <strong style={styles.statValue}>
              {data.forecast?.length || 0} days
            </strong>
          </div>
          <div style={styles.stat}>
            <span style={styles.statLabel}>Current Price</span>
            <strong style={styles.statValue}>
              RM {data.past_data?.[data.past_data.length - 1]?.price.toFixed(2)}
            </strong>
          </div>
        </div>
      </div>
    </div>
  )
}

const styles = {
  header: {
    marginBottom: '32px',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center'
  },
  exportBtn: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px'
  },
  chartContainer: {
    marginBottom: '32px'
  },
  stats: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
    gap: '24px',
    paddingTop: '24px',
    borderTop: '1px solid #e5e7eb'
  },
  stat: {
    textAlign: 'center',
    padding: '16px',
    background: '#f8fafc',
    borderRadius: '8px'
  },
  statLabel: {
    display: 'block',
    fontSize: '14px',
    color: '#6b7280',
    marginBottom: '8px'
  },
  statValue: {
    fontSize: '24px',
    color: '#1f2937'
  }
}

export default HistoricalTrends