import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './SidePanel.css';
import StateAnalytics from './StateAnalytics';


function SidePanel({
  states,
  counties,
  dataCenters,
  selectedState,
  selectedCounty,
  searchQuery,
  stats,
  selectedDC,
  analysis,
  onStateChange,
  onCountyChange,
  onSearchChange,
  onSelectDC,
  onLogout,
  totalCount
}) {

  var [viewMode, setViewMode] = useState('list');
  var nav = useNavigate();
  var MO = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  var MC = ['#60a5fa', '#60a5fa', '#4ade80', '#4ade80', '#4ade80', '#fbbf24', '#fbbf24', '#fbbf24', '#f97316', '#f97316', '#60a5fa', '#60a5fa'];

  var stateNames = {
    AL: 'Alabama', AK: 'Alaska', AZ: 'Arizona', AR: 'Arkansas', CA: 'California',
    CO: 'Colorado', CT: 'Connecticut', DC: 'DC', DE: 'Delaware', FL: 'Florida',
    GA: 'Georgia', HI: 'Hawaii', ID: 'Idaho', IL: 'Illinois', IN: 'Indiana',
    IA: 'Iowa', KS: 'Kansas', KY: 'Kentucky', LA: 'Louisiana', ME: 'Maine',
    MD: 'Maryland', MA: 'Massachusetts', MI: 'Michigan', MN: 'Minnesota',
    MS: 'Mississippi', MO: 'Missouri', MT: 'Montana', NE: 'Nebraska', NV: 'Nevada',
    NH: 'New Hampshire', NJ: 'New Jersey', NM: 'New Mexico', NY: 'New York',
    NC: 'North Carolina', ND: 'North Dakota', OH: 'Ohio', OK: 'Oklahoma',
    OR: 'Oregon', PA: 'Pennsylvania', PR: 'Puerto Rico', RI: 'Rhode Island',
    SC: 'South Carolina', SD: 'South Dakota', TN: 'Tennessee', TX: 'Texas',
    UT: 'Utah', VT: 'Vermont', VA: 'Virginia', WA: 'Washington', WV: 'West Virginia',
    WI: 'Wisconsin', WY: 'Wyoming'
  };

  function formatSqft(sqft) {
    if (sqft > 1000000) return (sqft / 1000000).toFixed(1) + 'M';
    return (sqft / 1000).toFixed(0) + 'K';
  }

  function getConfidence(dc) {
    var score = 50;
    if (dc.sqft > 500000) score += 20;
    else if (dc.sqft > 100000) score += 15;
    else if (dc.sqft > 50000) score += 8;
    if (dc.type === 'campus') score += 12;
    if (dc.operator && dc.operator.length > 2) score += 5;
    if (dc.sqft > 200000) score += 8;
    return Math.min(98, Math.max(35, score));
  }

  function getConfColor(conf) {
    if (conf >= 80) return '#16a34a';
    if (conf >= 60) return '#f59e0b';
    return '#dc2626';
  }

  function getViaColor(total) {
    if (total >= 75) return '#16a34a';
    if (total >= 50) return '#f59e0b';
    return '#dc2626';
  }

  // Build state options
  var stateOptions = [React.createElement('option', { key: 'all', value: '' }, 'All States')];
  if (states && states.length > 0) {
    states.forEach(function (s) {
      stateOptions.push(
        React.createElement('option', { key: s._id, value: s._id },
          (stateNames[s._id] || s._id) + ' (' + s.count + ')'
        )
      );
    });
  }

  // Build county options
  var countyOptions = [React.createElement('option', { key: 'all', value: '' }, 'All Counties')];
  if (counties && counties.length > 0) {
    counties.forEach(function (c) {
      countyOptions.push(
        React.createElement('option', { key: c._id, value: c._id },
          c._id + ' (' + c.count + ')'
        )
      );
    });
  }

  // Build list items
  var listContent;
  if (!dataCenters || dataCenters.length === 0) {
    listContent = React.createElement('div', { className: 'empty-state' },
      React.createElement('p', null, 'Select a state to view data centers or search by name above')
    );
  } else {
    var items = dataCenters.map(function (dc) {
      var conf = getConfidence(dc);
      var confColor = getConfColor(conf);
      var isSelected = selectedDC && selectedDC._id === dc._id;

      return React.createElement('div', {
        key: dc._id,
        className: 'item' + (isSelected ? ' selected' : ''),
        onClick: function () { onSelectDC(dc); }
      },
        React.createElement('div', { className: 'item-top' },
          React.createElement('div', {
            className: 'sqft-badge' + (dc.sqft > 100000 ? ' flagged' : '')
          }, formatSqft(dc.sqft)),
          React.createElement('div', { style: { flex: 1, minWidth: 0 } },
            React.createElement('div', { className: 'item-name' }, dc.name),
            React.createElement('div', { className: 'item-sub' },
              (dc.operator || 'Unknown') + ' | ' + dc.county
            )
          ),
          React.createElement('span', {
            className: 'confidence-tag',
            style: { background: confColor + '22', color: confColor }
          }, conf + '%'),
          dc.sqft > 100000 ? React.createElement('span', { className: 'flag-tag' }, 'FLAG') : null
        ),
        React.createElement('div', { className: 'item-stats' },
          React.createElement('span', null, dc.sqft.toLocaleString() + ' sqft'),
          React.createElement('span', null, dc.county),
          React.createElement('span', null, dc.state)
        )
      );
    });
    listContent = items;
  }

  // Detail panel
  var detailContent = null;
  if (selectedDC && analysis) {
    var femp = analysis.femp;
    var via = analysis.viability;
    var conf = analysis.confidence;
    var viaColor = getViaColor(via.total);

    var maxMonthly = Math.max.apply(null, femp.monthly);

    var bars = femp.monthly.map(function (g, i) {
      var height = maxMonthly > 0 ? (g / maxMonthly) * 40 : 0;
      return React.createElement('div', { key: i, className: 'chart-bar' },
        React.createElement('div', { className: 'chart-val' }, (g / 1000).toFixed(0) + 'K'),
        React.createElement('div', {
          className: 'chart-fill',
          style: { height: height + 'px', background: MC[i] }
        }),
        React.createElement('div', { className: 'chart-mo' }, MO[i])
      );
    });

    detailContent = React.createElement('div', { className: 'detail' },
      React.createElement('div', { className: 'detail-title' },
        'FEMP ANALYSIS - ' + selectedDC.name.slice(0, 35)
      ),
      React.createElement('div', { className: 'detail-grid' },
        React.createElement('div', { className: 'detail-stat' },
          React.createElement('div', { className: 'detail-stat-val', style: { color: '#1a6b4a' } },
            formatSqft(selectedDC.sqft) + ' sqft'
          ),
          React.createElement('div', { className: 'detail-stat-label' }, 'ROOF AREA')
        ),
        React.createElement('div', { className: 'detail-stat' },
          React.createElement('div', { className: 'detail-stat-val', style: { color: '#0ea5e9' } },
            (femp.annualHarvest / 1000).toFixed(0) + 'K gal'
          ),
          React.createElement('div', { className: 'detail-stat-label' }, 'ANNUAL HARVEST')
        ),
        React.createElement('div', { className: 'detail-stat' },
          React.createElement('div', { className: 'detail-stat-val', style: { color: '#f59e0b' } },
            '$' + femp.annualSavings.toLocaleString()
          ),
          React.createElement('div', { className: 'detail-stat-label' }, 'ANNUAL SAVINGS')
        ),
        React.createElement('div', { className: 'detail-stat' },
          React.createElement('div', { className: 'detail-stat-val', style: { color: viaColor } },
            via.total + '/100'
          ),
          React.createElement('div', { className: 'detail-stat-label' }, 'VIABILITY')
        )
      ),
      React.createElement('div', { className: 'chart' }, bars),
      React.createElement('div', { className: 'formula' },
        'FEMP: ' + selectedDC.sqft.toLocaleString() + ' sqft x ' +
        femp.rainfallTotal.toFixed(1) + 'in/yr x 0.80 eff x 0.62 conv = ' +
        femp.annualHarvest.toLocaleString() + ' gal/yr'
      ),
      React.createElement('div', { className: 'formula' },
        'Confidence: ' + conf + '% | Type: ' + selectedDC.type +
        ' | Cooling: ' + (selectedDC.sqft > 200000 ? 'Likely' : 'Unknown')
      ),
      React.createElement('div', { className: 'viability-section' },
        React.createElement('div', { style: { display: 'flex', justifyContent: 'space-between', alignItems: 'center' } },
          React.createElement('span', { className: 'viability-label' }, 'VIABILITY BREAKDOWN'),
          React.createElement('span', {
            style: { fontFamily: "'Space Mono', monospace", fontSize: '11px', fontWeight: 800, color: viaColor }
          }, via.total + '/100')
        ),
        React.createElement('div', { className: 'viability-bar' },
          React.createElement('div', {
            className: 'viability-fill',
            style: { width: via.total + '%', background: 'linear-gradient(90deg, ' + viaColor + ', #0ea5e9)' }
          })
        ),
        React.createElement('div', { className: 'viability-factors' },
          React.createElement('div', { className: 'viability-factor' },
            React.createElement('span', null, 'Physical'),
            React.createElement('span', { style: { fontWeight: 700 } }, via.physical + '/30')
          ),
          React.createElement('div', { className: 'viability-factor' },
            React.createElement('span', null, 'Rainfall'),
            React.createElement('span', { style: { fontWeight: 700 } }, via.rainfall + '/25')
          ),
          React.createElement('div', { className: 'viability-factor' },
            React.createElement('span', null, 'Financial'),
            React.createElement('span', { style: { fontWeight: 700 } }, via.financial + '/25')
          ),
          React.createElement('div', { className: 'viability-factor' },
            React.createElement('span', null, 'Regulatory'),
            React.createElement('span', { style: { fontWeight: 700 } }, via.regulatory + '/20')
          )
        )
      )
    );
  }

  var totalSqft = stats ? (stats.totalSqft / 1000000).toFixed(1) + 'M' : '0';

  return React.createElement('div', { id: 'panel' },
    React.createElement('div', { className: 'header' },
      React.createElement('div', { className: 'header-label' },
        React.createElement('span', null, 'SELECT LOCATION'),
        React.createElement('span', { className: 'header-count' }, totalCount + ' DATA CENTERS'),
        React.createElement('button', { className: 'logout-btn', onClick: onLogout }, 'LOGOUT')
      ),
      React.createElement('div', { className: 'filters' },
        React.createElement('select', {
          value: selectedState,
          onChange: function (e) { onStateChange(e.target.value); }
        }, stateOptions),
        React.createElement('select', {
          value: selectedCounty,
          disabled: !selectedState,
          onChange: function (e) { onCountyChange(e.target.value); }
        }, countyOptions)
      )
    ),
    selectedState ? React.createElement('div', { className: 'view-toggle' },
        React.createElement('button', {
          className: 'toggle-btn' + (viewMode === 'list' ? ' active' : ''),
          onClick: function () { setViewMode('list'); }
        }, 'LIST VIEW'),
        React.createElement('button', {
          className: 'toggle-btn',
          onClick: function () { nav('/analytics/' + selectedState); }
        }, 'STATE ANALYTICS')
      ) : null,
    React.createElement('div', { className: 'search-bar' },
      React.createElement('input', {
        className: 'search-input',
        type: 'text',
        placeholder: 'Search by name, operator, or county...',
        value: searchQuery,
        onChange: function (e) { onSearchChange(e.target.value); }
      })
    ),
    React.createElement('div', { className: 'stats' },
      React.createElement('div', { className: 'stat' },
        React.createElement('div', { className: 'stat-val', style: { color: '#16a34a' } }, stats ? stats.totalCenters : 0),
        React.createElement('div', { className: 'stat-label' }, 'CENTERS')
      ),
      React.createElement('div', { className: 'stat' },
        React.createElement('div', { className: 'stat-val', style: { color: '#dc2626' } }, stats ? stats.flaggedCount : 0),
        React.createElement('div', { className: 'stat-label' }, 'FLAGGED >100K')
      ),
      React.createElement('div', { className: 'stat' },
        React.createElement('div', { className: 'stat-val', style: { color: '#0ea5e9' } }, totalSqft),
        React.createElement('div', { className: 'stat-label' }, 'TOTAL SQFT')
      )
    ),
    viewMode === 'analytics' && selectedState && !selectedDC
      ? React.createElement(StateAnalytics, { state: selectedState, onSelectDC: function (dc) { setViewMode('list'); onSelectDC(dc); } })
      : React.createElement(React.Fragment, null,
          React.createElement('div', { className: 'list' }, listContent),
          detailContent
        )
  );
}

export default SidePanel;