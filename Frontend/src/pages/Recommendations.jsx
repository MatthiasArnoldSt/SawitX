import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { CheckCircle, XCircle, Clock, AlertCircle } from 'lucide-react'

const Recommendations = () => {
  const [recommendations, setRecommendations] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchRecommendations()
  }, [])

  const fetchRecommendations = async () => {
    try {
      setLoading(true)
      const response = await axios.get('/api/daily-recommendations')
      setRecommendations(response.data)
    } catch (error) {
      console.error('Error fetching recommendations:', error)
    } finally {
      setLoading(false)
    }
  }

  const getRecommendationIcon = (recommendation) => {
    const icons = {
      'HOLD': CheckCircle,
      'SELL': XCircle,
      'DELAY': Clock,
      'BUY': CheckCircle
    }
    return icons[recommendation] || AlertCircle
  }

  const getRecommendationColor = (color) => {
    const colors = {
      'green': '#10b981',
      'yellow': '#f59e0b',
      'red': '#ef4444',
      'orange': '#f97316'
    }
    return colors[color] || '#6b7280'
  }

  if (loading) {
    return <div className="loading">Loading recommendations...</div>
  }

  return (
    <div className="container">
      <div style={styles.header}>
        <h1>Daily Recommendations</h1>
        <p>Detailed trading advice with explanations</p>
      </div>

      <div className="card">
        <div style={styles.tableContainer}>
          <table style={styles.table}>
            <thead>
              <tr>
                <th style={styles.th}>Date</th>
                <th style={styles.th}>Forecast Price</th>
                <th style={styles.th}>Recommendation</th>
                <th style={styles.th}>Confidence</th>
                <th style={styles.th}>Explanation</th>
              </tr>
            </thead>
            <tbody>
              {recommendations.map((rec, index) => {
                const Icon = getRecommendationIcon(rec.recommendation)
                const color = getRecommendationColor(rec.color)
                
                return (
                  <tr key={index} style={styles.tr}>
                    <td style={styles.td}>
                      <strong>{rec.date}</strong>
                    </td>
                    <td style={styles.td}>
                      RM {rec.forecast_price.toFixed(2)}
                    </td>
                    <td style={styles.td}>
                      <div style={styles.recommendationCell}>
                        <Icon size={18} color={color} />
                        <span style={{ color, fontWeight: '600' }}>
                          {rec.recommendation}
                        </span>
                      </div>
                    </td>
                    <td style={styles.td}>
                      <div style={styles.confidence}>
                        <div style={styles.confidenceBar}>
                          <div 
                            style={{
                              ...styles.confidenceFill,
                              width: `${rec.confidence}%`,
                              background: color
                            }}
                          />
                        </div>
                        <span>{rec.confidence}%</span>
                      </div>
                    </td>
                    <td style={styles.td}>
                      {rec.explanation}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

const styles = {
  header: {
    marginBottom: '32px'
  },
  tableContainer: {
    overflowX: 'auto'
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse'
  },
  th: {
    padding: '12px 16px',
    textAlign: 'left',
    borderBottom: '2px solid #e5e7eb',
    fontWeight: '600',
    color: '#374151',
    background: '#f9fafb'
  },
  tr: {
    borderBottom: '1px solid #e5e7eb'
  },
  td: {
    padding: '16px',
    verticalAlign: 'top'
  },
  recommendationCell: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px'
  },
  confidence: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px'
  },
  confidenceBar: {
    width: '60px',
    height: '6px',
    background: '#e5e7eb',
    borderRadius: '3px',
    overflow: 'hidden'
  },
  confidenceFill: {
    height: '100%',
    borderRadius: '3px',
    transition: 'width 0.3s ease'
  }
}

export default Recommendations