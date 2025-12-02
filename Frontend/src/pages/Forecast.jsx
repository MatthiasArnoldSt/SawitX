import React, { useState, useEffect } from 'react'
import axios from 'axios'
import ForecastChart from '../components/ForecastChart'

const Forecast = () => {
  const [forecastData, setForecastData] = useState(null)
  const [selectedDays, setSelectedDays] = useState(14)
  const [loading, setLoading] = useState(false)

  const dayFilters = [1, 3, 7, 14, 28]

  useEffect(() => {
    fetchForecastData(selectedDays)
  }, [selectedDays])

  const fetchForecastData = async (days) => {
    try {
      setLoading(true)
      const response = await axios.get(`/api/forecast?days=${days}`)
      setForecastData(response.data)
    } catch (error) {
      console.error('Error fetching forecast data:', error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <div style={styles.header}>
        <h1>Price Forecast</h1>
        <p>Interactive FFB price predictions with confidence intervals</p>
      </div>

      <div className="card">
        <div style={styles.filterSection}>
          <div style={styles.filters}>
            <span style={styles.filterLabel}>Forecast Period:</span>
            {dayFilters.map(days => (
              <button
                key={days}
                onClick={() => setSelectedDays(days)}
                style={{
                  ...styles.filterBtn,
                  ...(selectedDays === days ? styles.filterBtnActive : {})
                }}
              >
                {days} Day{days !== 1 ? 's' : ''}
              </button>
            ))}
          </div>
        </div>

        {loading ? (
          <div className="loading">Loading forecast data...</div>
        ) : (
          forecastData && <ForecastChart data={forecastData} showTooltip />
        )}
      </div>
    </div>
  )
}

const styles = {
  header: {
    marginBottom: '32px'
  },
  filterSection: {
    marginBottom: '24px'
  },
  filters: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    flexWrap: 'wrap'
  },
  filterLabel: {
    fontWeight: '500',
    color: '#6b7280'
  },
  filterBtn: {
    padding: '6px 12px',
    border: '1px solid #d1d5db',
    borderRadius: '6px',
    background: 'white',
    cursor: 'pointer',
    transition: 'all 0.2s'
  },
  filterBtnActive: {
    background: '#3b82f6',
    color: 'white',
    borderColor: '#3b82f6'
  }
}

export default Forecast