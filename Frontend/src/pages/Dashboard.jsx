import React, { useState, useEffect } from 'react'
import { TrendingUp, TrendingDown, AlertCircle } from 'lucide-react'
import axios from 'axios'
import ForecastChart from '../components/ForecastChart'

const Dashboard = () => {
  const [forecastData, setForecastData] = useState(null)
  const [recommendation, setRecommendation] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchDashboardData()
  }, [])

  const fetchDashboardData = async () => {
    try {
      setLoading(true)
      const [forecastRes, recommendationRes] = await Promise.all([
        axios.get('/api/forecast?days=28'),
        axios.get('/api/recommendation')
      ])
      setForecastData(forecastRes.data)
      setRecommendation(recommendationRes.data)
    } catch (error) {
      console.error('Error fetching dashboard data:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return <div className="loading">Loading dashboard data...</div>
  }

  const getRecommendationColor = (color) => {
    const colors = {
      green: '#10b981',
      yellow: '#f59e0b',
      red: '#ef4444',
      orange: '#f97316'
    }
    return colors[color] || '#6b7280'
  }

  return (
    <div className="container">
      <div style={styles.header}>
        <h1>FFB Price Dashboard</h1>
        <p>Real-time forecasts and trading recommendations</p>
      </div>

      {/* Current Price & Recommendation Card */}
      <div className="card">
        <div style={styles.priceSection}>
          <div style={styles.priceInfo}>
            <h2 style={styles.currentPrice}>
              RM {recommendation?.current_price?.toFixed(2)}
            </h2>
            <div style={{
              ...styles.changeIndicator,
              color: recommendation?.price_change >= 0 ? '#10b981' : '#ef4444'
            }}>
              {recommendation?.price_change >= 0 ? 
                <TrendingUp size={16} /> : 
                <TrendingDown size={16} />
              }
              <span>{Math.abs(recommendation?.price_change || 0)}%</span>
            </div>
          </div>
          
          <div style={{
            ...styles.recommendationCard,
            borderLeft: `4px solid ${getRecommendationColor(recommendation?.color)}`
          }}>
            <h3 style={styles.recommendationTitle}>
              {recommendation?.recommendation}
            </h3>
            <p style={styles.recommendationExplanation}>
              {recommendation?.explanation}
            </p>
            <div style={styles.metrics}>
              <div style={styles.metric}>
                <span>Volatility</span>
                <strong>{recommendation?.volatility}%</strong>
              </div>
              <div style={styles.metric}>
                <span>Confidence</span>
                <strong>{recommendation?.confidence}%</strong>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Forecast Chart */}
      <div className="card">
        <h2 style={styles.chartTitle}>Price Forecast (Next 28 Days)</h2>
        {forecastData && <ForecastChart data={forecastData} />}
      </div>
    </div>
  )
}

const styles = {
  header: {
    marginBottom: '32px',
    textAlign: 'center'
  },
  priceSection: {
    display: 'grid',
    gridTemplateColumns: '1fr 2fr',
    gap: '32px',
    alignItems: 'center'
  },
  priceInfo: {
    textAlign: 'center'
  },
  currentPrice: {
    fontSize: '48px',
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: '8px'
  },
  changeIndicator: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '4px',
    fontWeight: '600',
    fontSize: '18px'
  },
  recommendationCard: {
    padding: '20px',
    background: '#f8fafc',
    borderRadius: '8px'
  },
  recommendationTitle: {
    fontSize: '24px',
    fontWeight: 'bold',
    marginBottom: '8px',
    color: '#1f2937'
  },
  recommendationExplanation: {
    color: '#6b7280',
    marginBottom: '16px',
    fontSize: '16px'
  },
  metrics: {
    display: 'flex',
    gap: '24px'
  },
  metric: {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px'
  },
  metricspan: {
    fontSize: '14px',
    color: '#6b7280'
  },
  chartTitle: {
    marginBottom: '24px',
    fontSize: '20px',
    fontWeight: '600',
    color: '#1f2937'
  }
}

export default Dashboard