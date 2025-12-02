import React from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

const ForecastChart = ({ data, showTooltip = false }) => {
  // Combine past and forecast data
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

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload
      return (
        <div style={{
          background: 'white',
          padding: '12px',
          border: '1px solid #e5e7eb',
          borderRadius: '6px',
          boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
        }}>
          <p style={{ fontWeight: '600', marginBottom: '4px' }}>{label}</p>
          <p style={{ color: '#3b82f6', margin: '2px 0' }}>
            Price: RM {data.price.toFixed(2)}
          </p>
          <p style={{ color: '#10b981', margin: '2px 0' }}>
            Type: {data.type}
          </p>
          {data.confidence && (
            <p style={{ color: '#f59e0b', margin: '2px 0' }}>
              Confidence: {data.confidence}%
            </p>
          )}
        </div>
      )
    }
    return null
  }

  return (
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
        <Tooltip content={showTooltip ? <CustomTooltip /> : null} />
        <Legend />
        <Line 
          type="monotone" 
          dataKey="price" 
          stroke="#3b82f6" 
          strokeWidth={2}
          dot={false}
          name="FFB Price"
          strokeDasharray={chartData.length > 0 && chartData[chartData.length - 1].type === 'Forecast' ? '5 5' : '0'}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}

export default ForecastChart