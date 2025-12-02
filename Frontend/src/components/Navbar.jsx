import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import { BarChart3, TrendingUp, Clock, Home } from 'lucide-react'

const Navbar = () => {
  const location = useLocation()

  const navItems = [
    { path: '/', icon: Home, label: 'Dashboard' },
    { path: '/forecast', icon: TrendingUp, label: 'Forecast' },
    { path: '/recommendations', icon: BarChart3, label: 'Recommendations' },
    { path: '/historical-trends', icon: Clock, label: 'Historical Trends' },
  ]

  return (
    <nav style={styles.navbar}>
      <div className="container">
        <div style={styles.navContent}>
          <div style={styles.logo}>
            <BarChart3 size={24} />
            <span style={styles.logoText}>FFB Forecast</span>
          </div>
          <div style={styles.navLinks}>
            {navItems.map((item) => {
              const Icon = item.icon
              const isActive = location.pathname === item.path
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  style={{
                    ...styles.navLink,
                    ...(isActive ? styles.navLinkActive : {})
                  }}
                >
                  <Icon size={18} />
                  <span>{item.label}</span>
                </Link>
              )
            })}
          </div>
        </div>
      </div>
    </nav>
  )
}

const styles = {
  navbar: {
    background: 'white',
    borderBottom: '1px solid #e5e5e5',
    padding: '16px 0',
    marginBottom: '32px'
  },
  navContent: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center'
  },
  logo: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontWeight: 'bold',
    fontSize: '20px',
    color: '#3b82f6'
  },
  logoText: {
    fontWeight: 'bold'
  },
  navLinks: {
    display: 'flex',
    gap: '24px'
  },
  navLink: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    textDecoration: 'none',
    color: '#666',
    padding: '8px 12px',
    borderRadius: '6px',
    transition: 'all 0.2s',
    fontWeight: '500'
  },
  navLinkActive: {
    background: '#3b82f6',
    color: 'white'
  }
}

export default Navbar