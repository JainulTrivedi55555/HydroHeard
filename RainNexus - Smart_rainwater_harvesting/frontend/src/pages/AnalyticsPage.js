import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, AreaChart, Area } from 'recharts';
import { getStateAnalytics } from '../services/api';
import './AnalyticsPage.css';

var STATE_NAMES = {
  AL:'Alabama',AK:'Alaska',AZ:'Arizona',AR:'Arkansas',CA:'California',
  CO:'Colorado',CT:'Connecticut',DC:'DC',DE:'Delaware',FL:'Florida',
  GA:'Georgia',HI:'Hawaii',ID:'Idaho',IL:'Illinois',IN:'Indiana',
  IA:'Iowa',KS:'Kansas',KY:'Kentucky',LA:'Louisiana',ME:'Maine',
  MD:'Maryland',MA:'Massachusetts',MI:'Michigan',MN:'Minnesota',
  MS:'Mississippi',MO:'Missouri',MT:'Montana',NE:'Nebraska',NV:'Nevada',
  NH:'New Hampshire',NJ:'New Jersey',NM:'New Mexico',NY:'New York',
  NC:'North Carolina',ND:'North Dakota',OH:'Ohio',OK:'Oklahoma',
  OR:'Oregon',PA:'Pennsylvania',PR:'Puerto Rico',RI:'Rhode Island',
  SC:'South Carolina',SD:'South Dakota',TN:'Tennessee',TX:'Texas',
  UT:'Utah',VT:'Vermont',VA:'Virginia',WA:'Washington',WV:'West Virginia',
  WI:'Wisconsin',WY:'Wyoming'
};

var MO = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
var PIE_COLORS = ['#0ea5e9','#16a34a','#f59e0b','#8b5cf6','#ec4899','#06b6d4','#84cc16','#f97316','#6366f1','#14b8a6'];

function AnalyticsPage() {
  var params = useParams();
  var navigate = useNavigate();
  var state = params.state;

  var [data, setData] = useState(null);
  var [loading, setLoading] = useState(true);

  useEffect(function () {
    if (!state) return;
    setLoading(true);
    getStateAnalytics(state).then(function (res) {
      setData(res.data);
      setLoading(false);
    }).catch(function () {
      setLoading(false);
    });
  }, [state]);

  function formatNum(n) {
    if (n >= 1000000000) return (n / 1000000000).toFixed(1) + 'B';
    if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
    if (n >= 1000) return (n / 1000).toFixed(0) + 'K';
    return n.toString();
  }

  function handleProspectClick(dc) {
    navigate('/dashboard?state=' + state + '&dc=' + dc._id);
  }

  if (loading) {
    return React.createElement('div', { className: 'ap-loading' },
      React.createElement('div', { className: 'ap-loading-text' }, 'Loading analytics for ' + (STATE_NAMES[state] || state) + '...')
    );
  }

  if (!data) {
    return React.createElement('div', { className: 'ap-loading' },
      React.createElement('div', { className: 'ap-loading-text' }, 'No data available'),
      React.createElement('button', { className: 'ap-back-btn', onClick: function () { navigate('/dashboard'); } }, 'Back to Dashboard')
    );
  }

  var ov = data.overview;
  var rf = data.rainfall;
  var costs = data.costs;

  var gradeColor = '#16a34a';
  if (ov.grade.startsWith('B')) gradeColor = '#f59e0b';
  if (ov.grade.startsWith('C')) gradeColor = '#dc2626';

  var rainfallChartData = rf.monthly.map(function (val, i) {
    return { month: MO[i], rainfall: parseFloat(val.toFixed(2)) };
  });

  var harvestChartData = data.monthlyHarvest.map(function (val, i) {
    return { month: MO[i], harvest: Math.round(val / 1000000) };
  });

  // Cumulative harvest
  var cumulative = 0;
  var cumulativeData = data.monthlyHarvest.map(function (val, i) {
    cumulative += val;
    return { month: MO[i], cumulative: Math.round(cumulative / 1000000) };
  });

  var dist = data.sqftDistribution;
  var distChartData = [
    { range: '<50K', count: dist.under50k, fill: '#94a3b8' },
    { range: '50-100K', count: dist.from50kTo100k, fill: '#60a5fa' },
    { range: '100-500K', count: dist.from100kTo500k, fill: '#16a34a' },
    { range: '500K-1M', count: dist.from500kTo1m, fill: '#f59e0b' },
    { range: '>1M', count: dist.over1m, fill: '#dc2626' }
  ];

  var operatorPieData = data.operatorBreakdown.map(function (op) {
    return { name: op.operator, value: op.count };
  });

  return React.createElement('div', { className: 'ap-page' },

    // Top Bar
    React.createElement('div', { className: 'ap-topbar' },
      React.createElement('button', {
        className: 'ap-back-btn',
        onClick: function () { navigate('/dashboard?state=' + state); }
      },
        React.createElement('span', { className: 'ap-back-arrow' }, '\u2190'),
        ' Back to Dashboard'
      ),
      React.createElement('div', { className: 'ap-topbar-logo' },
        React.createElement('svg', { width: 18, height: 18, viewBox: '0 0 32 32', fill: 'none' },
          React.createElement('path', { d: 'M16 3C16 3 6 15 6 21C6 26.5 10.5 29 16 29C21.5 29 26 26.5 26 21C26 15 16 3 16 3Z', fill: '#0ea5e9', opacity: '.3', stroke: '#0ea5e9', strokeWidth: '2' })
        ),
        React.createElement('span', null, 'RAINUSE NEXUS')
      ),
      React.createElement('div', { className: 'ap-topbar-right' },
        React.createElement('span', { className: 'ap-chip' }, 'GEMINI AI POWERED'),
        React.createElement('span', { className: 'ap-chip purple' }, 'SOLANA VERIFIED')
      )
    ),

    // Content
    React.createElement('div', { className: 'ap-content' },

      // Header Section
      React.createElement('div', { className: 'ap-hero' },
        React.createElement('div', { className: 'ap-hero-left' },
          React.createElement('h1', { className: 'ap-title' }, STATE_NAMES[state] || state),
          React.createElement('p', { className: 'ap-subtitle' }, 'Water Opportunity Analysis Report'),
          React.createElement('div', { className: 'ap-hero-tags' },
            React.createElement('span', { className: 'ap-tag' }, ov.totalCenters + ' Data Centers'),
            React.createElement('span', { className: 'ap-tag green' }, ov.flaggedCount + ' Flagged >100K sqft'),
            React.createElement('span', { className: 'ap-tag blue' }, rf.annualTotal.toFixed(1) + ' in/yr rainfall')
          )
        ),
        React.createElement('div', {
          className: 'ap-grade-box',
          style: { borderColor: gradeColor + '60', background: gradeColor + '10' }
        },
          React.createElement('div', { className: 'ap-grade-label' }, 'WATER OPPORTUNITY'),
          React.createElement('div', { className: 'ap-grade-value', style: { color: gradeColor } }, ov.grade),
          React.createElement('div', { className: 'ap-grade-label' }, 'GRADE')
        )
      ),

      // Key Metrics Row
      React.createElement('div', { className: 'ap-metrics-row' },
        React.createElement('div', { className: 'ap-metric-card' },
          React.createElement('div', { className: 'ap-mc-icon', style: { background: '#16a34a20', color: '#16a34a' } }, '\u2302'),
          React.createElement('div', { className: 'ap-mc-content' },
            React.createElement('div', { className: 'ap-mc-val' }, ov.totalCenters.toLocaleString()),
            React.createElement('div', { className: 'ap-mc-label' }, 'DATA CENTERS')
          )
        ),
        React.createElement('div', { className: 'ap-metric-card' },
          React.createElement('div', { className: 'ap-mc-icon', style: { background: '#0ea5e920', color: '#0ea5e9' } }, '\u25A3'),
          React.createElement('div', { className: 'ap-mc-content' },
            React.createElement('div', { className: 'ap-mc-val' }, formatNum(ov.totalSqft) + ' sqft'),
            React.createElement('div', { className: 'ap-mc-label' }, 'TOTAL ROOF AREA')
          )
        ),
        React.createElement('div', { className: 'ap-metric-card' },
          React.createElement('div', { className: 'ap-mc-icon', style: { background: '#4ade8020', color: '#4ade80' } }, '\u2248'),
          React.createElement('div', { className: 'ap-mc-content' },
            React.createElement('div', { className: 'ap-mc-val' }, formatNum(ov.totalHarvest) + ' gal'),
            React.createElement('div', { className: 'ap-mc-label' }, 'ANNUAL HARVEST POTENTIAL')
          )
        ),
        React.createElement('div', { className: 'ap-metric-card highlight' },
          React.createElement('div', { className: 'ap-mc-icon', style: { background: '#f59e0b20', color: '#f59e0b' } }, '$'),
          React.createElement('div', { className: 'ap-mc-content' },
            React.createElement('div', { className: 'ap-mc-val' }, '$' + formatNum(ov.totalSavings)),
            React.createElement('div', { className: 'ap-mc-label' }, 'ANNUAL SAVINGS POTENTIAL')
          )
        )
      ),

      // Comparison Cards
      React.createElement('div', { className: 'ap-compare-row' },
        React.createElement('div', { className: 'ap-compare-card' },
          React.createElement('div', { className: 'ap-cc-header' }, 'ANNUAL RAINFALL'),
          React.createElement('div', { className: 'ap-cc-main' },
            React.createElement('span', { className: 'ap-cc-val' }, rf.annualTotal.toFixed(1)),
            React.createElement('span', { className: 'ap-cc-unit' }, 'in/yr')
          ),
          React.createElement('div', { className: 'ap-cc-bar-bg' },
            React.createElement('div', { className: 'ap-cc-bar-fill', style: { width: Math.min(100, (rf.annualTotal / 50) * 100) + '%', background: '#0ea5e9' } }),
            React.createElement('div', { className: 'ap-cc-bar-marker', style: { left: Math.min(100, (rf.nationalAvgRain / 50) * 100) + '%' } })
          ),
          React.createElement('div', { className: 'ap-cc-footer' },
            React.createElement('span', null, 'National avg: ' + rf.nationalAvgRain + ' in/yr'),
            React.createElement('span', {
              style: { color: rf.annualTotal >= rf.nationalAvgRain ? '#16a34a' : '#dc2626', fontWeight: 700 }
            }, rf.annualTotal >= rf.nationalAvgRain
              ? '+' + (rf.annualTotal - rf.nationalAvgRain).toFixed(1) + ' above'
              : (rf.annualTotal - rf.nationalAvgRain).toFixed(1) + ' below'
            )
          )
        ),
        React.createElement('div', { className: 'ap-compare-card' },
          React.createElement('div', { className: 'ap-cc-header' }, 'WATER UTILITY COST'),
          React.createElement('div', { className: 'ap-cc-main' },
            React.createElement('span', { className: 'ap-cc-val' }, '$' + costs.costPerThousand),
            React.createElement('span', { className: 'ap-cc-unit' }, '/1K gal')
          ),
          React.createElement('div', { className: 'ap-cc-bar-bg' },
            React.createElement('div', { className: 'ap-cc-bar-fill', style: { width: Math.min(100, (costs.costPerThousand / 22) * 100) + '%', background: '#f59e0b' } }),
            React.createElement('div', { className: 'ap-cc-bar-marker', style: { left: Math.min(100, (costs.nationalAvgCost / 22) * 100) + '%' } })
          ),
          React.createElement('div', { className: 'ap-cc-footer' },
            React.createElement('span', null, 'National avg: $' + costs.nationalAvgCost + '/1K gal'),
            React.createElement('span', {
              style: { color: costs.costPerThousand >= costs.nationalAvgCost ? '#16a34a' : '#dc2626', fontWeight: 700 }
            }, costs.costPerThousand >= costs.nationalAvgCost
              ? '+$' + (costs.costPerThousand - costs.nationalAvgCost).toFixed(1) + ' higher ROI'
              : '-$' + (costs.nationalAvgCost - costs.costPerThousand).toFixed(1) + ' lower ROI'
            )
          )
        )
      ),

      // Charts Row
      React.createElement('div', { className: 'ap-charts-row' },
        React.createElement('div', { className: 'ap-chart-card' },
          React.createElement('div', { className: 'ap-chart-header' }, 'MONTHLY RAINFALL (INCHES)'),
          React.createElement(ResponsiveContainer, { width: '100%', height: 180 },
            React.createElement(BarChart, { data: rainfallChartData },
              React.createElement(XAxis, { dataKey: 'month', tick: { fontSize: 10, fill: '#6b8a72' } }),
              React.createElement(YAxis, { tick: { fontSize: 10, fill: '#6b8a72' }, width: 30 }),
              React.createElement(Tooltip, { contentStyle: { fontSize: 12, fontFamily: 'Space Mono', borderRadius: 8 } }),
              React.createElement(Bar, { dataKey: 'rainfall', fill: '#0ea5e9', radius: [3, 3, 0, 0] })
            )
          )
        ),
        React.createElement('div', { className: 'ap-chart-card' },
          React.createElement('div', { className: 'ap-chart-header' }, 'CUMULATIVE HARVEST (MILLION GAL)'),
          React.createElement(ResponsiveContainer, { width: '100%', height: 180 },
            React.createElement(AreaChart, { data: cumulativeData },
              React.createElement(XAxis, { dataKey: 'month', tick: { fontSize: 10, fill: '#6b8a72' } }),
              React.createElement(YAxis, { tick: { fontSize: 10, fill: '#6b8a72' }, width: 30 }),
              React.createElement(Tooltip, { contentStyle: { fontSize: 12, fontFamily: 'Space Mono', borderRadius: 8 } }),
              React.createElement(Area, { dataKey: 'cumulative', fill: '#16a34a30', stroke: '#16a34a', strokeWidth: 2 })
            )
          )
        )
      ),

      // Second Charts Row
      React.createElement('div', { className: 'ap-charts-row' },
        React.createElement('div', { className: 'ap-chart-card' },
          React.createElement('div', { className: 'ap-chart-header' }, 'OPERATOR BREAKDOWN'),
          React.createElement(ResponsiveContainer, { width: '100%', height: 200 },
            React.createElement(PieChart, null,
              React.createElement(Pie, {
                data: operatorPieData,
                cx: '50%',
                cy: '50%',
                outerRadius: 70,
                innerRadius: 35,
                dataKey: 'value',
                paddingAngle: 2
              },
                operatorPieData.map(function (entry, i) {
                  return React.createElement(Cell, { key: i, fill: PIE_COLORS[i % PIE_COLORS.length] });
                })
              ),
              React.createElement(Tooltip, { contentStyle: { fontSize: 11, fontFamily: 'Space Mono', borderRadius: 8 } })
            )
          ),
          React.createElement('div', { className: 'ap-legend' },
            operatorPieData.slice(0, 6).map(function (op, i) {
              return React.createElement('span', { key: i, className: 'ap-legend-item' },
                React.createElement('span', { className: 'ap-legend-dot', style: { background: PIE_COLORS[i % PIE_COLORS.length] } }),
                op.name.slice(0, 15) + ' (' + op.value + ')'
              );
            })
          )
        ),
        React.createElement('div', { className: 'ap-chart-card' },
          React.createElement('div', { className: 'ap-chart-header' }, 'FACILITY SIZE DISTRIBUTION'),
          React.createElement(ResponsiveContainer, { width: '100%', height: 200 },
            React.createElement(BarChart, { data: distChartData, layout: 'vertical' },
              React.createElement(XAxis, { type: 'number', tick: { fontSize: 10, fill: '#6b8a72' } }),
              React.createElement(YAxis, { dataKey: 'range', type: 'category', tick: { fontSize: 10, fill: '#6b8a72' }, width: 60 }),
              React.createElement(Tooltip, { contentStyle: { fontSize: 12, fontFamily: 'Space Mono', borderRadius: 8 } }),
              React.createElement(Bar, { dataKey: 'count', fill: '#8b5cf6', radius: [0, 3, 3, 0] })
            )
          )
        )
      ),

      // Top Prospects
      React.createElement('div', { className: 'ap-prospects-section' },
        React.createElement('div', { className: 'ap-chart-header' }, 'TOP 5 HIGHEST-VALUE PROSPECTS'),
        React.createElement('div', { className: 'ap-prospects-table' },
          React.createElement('div', { className: 'ap-pt-header' },
            React.createElement('span', null, 'RANK'),
            React.createElement('span', null, 'FACILITY'),
            React.createElement('span', null, 'OPERATOR'),
            React.createElement('span', null, 'COUNTY'),
            React.createElement('span', null, 'ROOF SQFT'),
            React.createElement('span', null, 'HARVEST/YR'),
            React.createElement('span', null, 'SAVINGS/YR')
          ),
          data.topProspects.map(function (dc, i) {
            return React.createElement('div', {
              key: dc._id,
              className: 'ap-pt-row',
              onClick: function () { handleProspectClick(dc); }
            },
              React.createElement('span', { className: 'ap-pt-rank' }, '#' + (i + 1)),
              React.createElement('span', { className: 'ap-pt-name' }, dc.name),
              React.createElement('span', null, dc.operator || 'Unknown'),
              React.createElement('span', null, dc.county),
              React.createElement('span', null, formatNum(dc.sqft)),
              React.createElement('span', { style: { color: '#0ea5e9' } }, formatNum(dc.annualHarvest) + ' gal'),
              React.createElement('span', { style: { color: '#f59e0b', fontWeight: 700 } }, '$' + formatNum(dc.annualSavings))
            );
          })
        )
      ),

      // County Breakdown
      React.createElement('div', { className: 'ap-county-section' },
        React.createElement('div', { className: 'ap-chart-header' }, 'COUNTY OPPORTUNITY RANKING'),
        React.createElement('div', { className: 'ap-county-grid' },
          data.countyBreakdown.map(function (c, i) {
            var maxSavings = data.countyBreakdown[0].totalSavings;
            var pct = maxSavings > 0 ? (c.totalSavings / maxSavings) * 100 : 0;
            return React.createElement('div', { key: i, className: 'ap-county-card' },
              React.createElement('div', { className: 'ap-county-top' },
                React.createElement('span', { className: 'ap-county-name' }, c.county),
                React.createElement('span', { className: 'ap-county-savings' }, '$' + formatNum(c.totalSavings) + '/yr')
              ),
              React.createElement('div', { className: 'ap-county-bar' },
                React.createElement('div', { className: 'ap-county-fill', style: { width: pct + '%' } })
              ),
              React.createElement('div', { className: 'ap-county-bottom' },
                React.createElement('span', null, c.count + ' centers'),
                React.createElement('span', null, formatNum(c.totalSqft) + ' sqft')
              )
            );
          })
        )
      ),

      // Footer
      React.createElement('div', { className: 'ap-footer' },
        React.createElement('span', null, 'RAINUSE NEXUS by Grundfos | Data: FEMP Calculator, EPA Water Data'),
        React.createElement('span', null, 'Analysis powered by Gemini AI | Verified on Solana')
      )
    )
  );
}

export default AnalyticsPage;