import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { getStateAnalytics } from '../services/api';
import './StateAnalytics.css';

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

function StateAnalytics({ state, onSelectDC }) {
  var [data, setData] = useState(null);
  var [loading, setLoading] = useState(true);
  var [error, setError] = useState(null);

  useEffect(function () {
    if (!state) return;
    setLoading(true);
    setError(null);

    getStateAnalytics(state).then(function (res) {
      setData(res.data);
      setLoading(false);
    }).catch(function (err) {
      setError('Failed to load analytics');
      setLoading(false);
    });
  }, [state]);

  if (loading) {
    return React.createElement('div', { className: 'sa-loading' }, 'Loading state analytics...');
  }

  if (error || !data) {
    return React.createElement('div', { className: 'sa-loading' }, error || 'No data available');
  }

  var ov = data.overview;
  var rf = data.rainfall;
  var costs = data.costs;

  // Grade color
  var gradeColor = '#16a34a';
  if (ov.grade.startsWith('B')) gradeColor = '#f59e0b';
  if (ov.grade.startsWith('C')) gradeColor = '#dc2626';

  // Rainfall chart data
  var rainfallChartData = rf.monthly.map(function (val, i) {
    return { month: MO[i], rainfall: parseFloat(val.toFixed(2)) };
  });

  // Monthly harvest chart data
  var harvestChartData = data.monthlyHarvest.map(function (val, i) {
    return { month: MO[i], harvest: Math.round(val / 1000) };
  });

  // Sqft distribution chart data
  var dist = data.sqftDistribution;
  var distChartData = [
    { range: '<50K', count: dist.under50k },
    { range: '50-100K', count: dist.from50kTo100k },
    { range: '100-500K', count: dist.from100kTo500k },
    { range: '500K-1M', count: dist.from500kTo1m },
    { range: '>1M', count: dist.over1m }
  ];

  // Operator pie data
  var operatorPieData = data.operatorBreakdown.map(function (op) {
    return { name: op.operator, value: op.count };
  });

  function formatNum(n) {
    if (n >= 1000000000) return (n / 1000000000).toFixed(1) + 'B';
    if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
    if (n >= 1000) return (n / 1000).toFixed(0) + 'K';
    return n.toString();
  }

  return React.createElement('div', { className: 'state-analytics' },

    // Section 1: Header
    React.createElement('div', { className: 'sa-header' },
      React.createElement('div', { className: 'sa-header-left' },
        React.createElement('h2', { className: 'sa-state-name' }, STATE_NAMES[state] || state),
        React.createElement('span', { className: 'sa-subtitle' }, 'Water Opportunity Analysis')
      ),
      React.createElement('div', {
        className: 'sa-grade',
        style: { background: gradeColor + '20', color: gradeColor, borderColor: gradeColor + '40' }
      },
        React.createElement('span', { className: 'sa-grade-label' }, 'GRADE'),
        React.createElement('span', { className: 'sa-grade-value' }, ov.grade)
      )
    ),

    // Section 2: Key Metrics
    React.createElement('div', { className: 'sa-metrics' },
      React.createElement('div', { className: 'sa-metric' },
        React.createElement('div', { className: 'sa-metric-val', style: { color: '#16a34a' } }, ov.totalCenters),
        React.createElement('div', { className: 'sa-metric-label' }, 'DATA CENTERS')
      ),
      React.createElement('div', { className: 'sa-metric' },
        React.createElement('div', { className: 'sa-metric-val', style: { color: '#0ea5e9' } }, formatNum(ov.totalSqft)),
        React.createElement('div', { className: 'sa-metric-label' }, 'TOTAL SQFT')
      ),
      React.createElement('div', { className: 'sa-metric' },
        React.createElement('div', { className: 'sa-metric-val', style: { color: '#4ade80' } }, formatNum(ov.totalHarvest) + ' gal'),
        React.createElement('div', { className: 'sa-metric-label' }, 'HARVEST/YR')
      ),
      React.createElement('div', { className: 'sa-metric' },
        React.createElement('div', { className: 'sa-metric-val', style: { color: '#f59e0b' } }, '$' + formatNum(ov.totalSavings)),
        React.createElement('div', { className: 'sa-metric-label' }, 'SAVINGS/YR')
      ),
      React.createElement('div', { className: 'sa-metric' },
        React.createElement('div', { className: 'sa-metric-val', style: { color: '#dc2626' } }, ov.flaggedCount),
        React.createElement('div', { className: 'sa-metric-label' }, 'FLAGGED >100K')
      )
    ),

    // Section 3: Comparison bars
    React.createElement('div', { className: 'sa-comparisons' },
      React.createElement('div', { className: 'sa-comparison' },
        React.createElement('span', { className: 'sa-comp-label' }, 'Avg Rainfall'),
        React.createElement('span', { className: 'sa-comp-val' }, rf.annualTotal.toFixed(1) + ' in/yr'),
        React.createElement('span', { className: 'sa-comp-vs' },
          rf.annualTotal > rf.nationalAvgRain
            ? '+' + (rf.annualTotal - rf.nationalAvgRain).toFixed(1) + ' vs national'
            : (rf.annualTotal - rf.nationalAvgRain).toFixed(1) + ' vs national'
        )
      ),
      React.createElement('div', { className: 'sa-comparison' },
        React.createElement('span', { className: 'sa-comp-label' }, 'Water Cost'),
        React.createElement('span', { className: 'sa-comp-val' }, '$' + costs.costPerThousand + '/1K gal'),
        React.createElement('span', { className: 'sa-comp-vs' },
          costs.costPerThousand > costs.nationalAvgCost
            ? '+$' + (costs.costPerThousand - costs.nationalAvgCost).toFixed(1) + ' vs national'
            : '-$' + (costs.nationalAvgCost - costs.costPerThousand).toFixed(1) + ' vs national'
        )
      )
    ),

    // Section 4: Monthly Rainfall Chart
    React.createElement('div', { className: 'sa-chart-section' },
      React.createElement('div', { className: 'sa-chart-title' }, 'MONTHLY RAINFALL (INCHES)'),
      React.createElement(ResponsiveContainer, { width: '100%', height: 120 },
        React.createElement(BarChart, { data: rainfallChartData },
          React.createElement(XAxis, { dataKey: 'month', tick: { fontSize: 8, fill: '#6b8a72' } }),
          React.createElement(YAxis, { tick: { fontSize: 8, fill: '#6b8a72' }, width: 25 }),
          React.createElement(Tooltip, { contentStyle: { fontSize: 11, fontFamily: 'Space Mono' } }),
          React.createElement(Bar, { dataKey: 'rainfall', fill: '#0ea5e9', radius: [2, 2, 0, 0] })
        )
      )
    ),

    // Section 5: Monthly Harvest Chart
    React.createElement('div', { className: 'sa-chart-section' },
      React.createElement('div', { className: 'sa-chart-title' }, 'MONTHLY HARVEST POTENTIAL (THOUSAND GAL)'),
      React.createElement(ResponsiveContainer, { width: '100%', height: 120 },
        React.createElement(BarChart, { data: harvestChartData },
          React.createElement(XAxis, { dataKey: 'month', tick: { fontSize: 8, fill: '#6b8a72' } }),
          React.createElement(YAxis, { tick: { fontSize: 8, fill: '#6b8a72' }, width: 35 }),
          React.createElement(Tooltip, { contentStyle: { fontSize: 11, fontFamily: 'Space Mono' } }),
          React.createElement(Bar, { dataKey: 'harvest', fill: '#16a34a', radius: [2, 2, 0, 0] })
        )
      )
    ),

    // Section 6: Top 5 Prospects
    React.createElement('div', { className: 'sa-section' },
      React.createElement('div', { className: 'sa-chart-title' }, 'TOP 5 PROSPECTS'),
      React.createElement('div', { className: 'sa-prospects' },
        data.topProspects.map(function (dc, i) {
          return React.createElement('div', {
            key: dc._id,
            className: 'sa-prospect',
            onClick: function () { onSelectDC(dc); }
          },
            React.createElement('span', { className: 'sa-prospect-rank' }, '#' + (i + 1)),
            React.createElement('div', { className: 'sa-prospect-info' },
              React.createElement('div', { className: 'sa-prospect-name' }, dc.name),
              React.createElement('div', { className: 'sa-prospect-sub' },
                (dc.operator || 'Unknown') + ' | ' + dc.county
              )
            ),
            React.createElement('div', { className: 'sa-prospect-stats' },
              React.createElement('div', { className: 'sa-prospect-stat' },
                React.createElement('span', { style: { color: '#0ea5e9' } }, formatNum(dc.sqft) + ' sqft')
              ),
              React.createElement('div', { className: 'sa-prospect-stat' },
                React.createElement('span', { style: { color: '#f59e0b', fontWeight: 700 } }, '$' + formatNum(dc.annualSavings) + '/yr')
              )
            )
          );
        })
      )
    ),

    // Section 7: Operator Breakdown + Sqft Distribution side by side
    React.createElement('div', { className: 'sa-charts-row' },
      React.createElement('div', { className: 'sa-chart-half' },
        React.createElement('div', { className: 'sa-chart-title' }, 'OPERATORS'),
        React.createElement(ResponsiveContainer, { width: '100%', height: 150 },
          React.createElement(PieChart, null,
            React.createElement(Pie, {
              data: operatorPieData,
              cx: '50%',
              cy: '50%',
              outerRadius: 55,
              dataKey: 'value',
              label: function (entry) { return entry.name.slice(0, 10); },
              labelLine: false
            },
              operatorPieData.map(function (entry, i) {
                return React.createElement(Cell, {
                  key: i,
                  fill: PIE_COLORS[i % PIE_COLORS.length]
                });
              })
            ),
            React.createElement(Tooltip, { contentStyle: { fontSize: 10, fontFamily: 'Space Mono' } })
          )
        )
      ),
      React.createElement('div', { className: 'sa-chart-half' },
        React.createElement('div', { className: 'sa-chart-title' }, 'SIZE DISTRIBUTION'),
        React.createElement(ResponsiveContainer, { width: '100%', height: 150 },
          React.createElement(BarChart, { data: distChartData },
            React.createElement(XAxis, { dataKey: 'range', tick: { fontSize: 8, fill: '#6b8a72' } }),
            React.createElement(YAxis, { tick: { fontSize: 8, fill: '#6b8a72' }, width: 25 }),
            React.createElement(Tooltip, { contentStyle: { fontSize: 11, fontFamily: 'Space Mono' } }),
            React.createElement(Bar, { dataKey: 'count', fill: '#8b5cf6', radius: [2, 2, 0, 0] })
          )
        )
      )
    ),

    // Section 8: County Breakdown
    React.createElement('div', { className: 'sa-section' },
      React.createElement('div', { className: 'sa-chart-title' }, 'TOP COUNTIES BY OPPORTUNITY'),
      React.createElement('div', { className: 'sa-counties' },
        data.countyBreakdown.map(function (c, i) {
          var maxSavings = data.countyBreakdown[0].totalSavings;
          var barWidth = maxSavings > 0 ? (c.totalSavings / maxSavings) * 100 : 0;

          return React.createElement('div', { key: i, className: 'sa-county-row' },
            React.createElement('span', { className: 'sa-county-name' }, c.county),
            React.createElement('span', { className: 'sa-county-count' }, c.count + ' centers'),
            React.createElement('div', { className: 'sa-county-bar-bg' },
              React.createElement('div', {
                className: 'sa-county-bar-fill',
                style: { width: barWidth + '%' }
              })
            ),
            React.createElement('span', { className: 'sa-county-val' }, '$' + formatNum(c.totalSavings))
          );
        })
      )
    )
  );
}

export default StateAnalytics;