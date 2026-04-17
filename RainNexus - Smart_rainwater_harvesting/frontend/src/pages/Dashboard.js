import React, { useState, useEffect, useCallback } from 'react';
import { useAuth } from '../context/AuthContext';
import { useNavigate } from 'react-router-dom';
import MapView from '../components/MapView';
import SidePanel from '../components/SidePanel';
import { useAuth0 } from '@auth0/auth0-react';
import {
  getDataCenters,
  getDataCenterById,
  getStates,
  getCounties,
  getStats
} from '../services/api';
import './Dashboard.css';

function Dashboard() {
  var auth = useAuth();
  var navigate = useNavigate();
  var auth0 = useAuth0();

  var [states, setStates] = useState([]);
  var [counties, setCounties] = useState([]);
  var [dataCenters, setDataCenters] = useState([]);
  var [selectedState, setSelectedState] = useState('');
  var [selectedCounty, setSelectedCounty] = useState('');
  var [searchQuery, setSearchQuery] = useState('');
  var [stats, setStats] = useState(null);
  var [selectedDC, setSelectedDC] = useState(null);
  var [analysis, setAnalysis] = useState(null);
  var [totalCount, setTotalCount] = useState(0);
  var [loading, setLoading] = useState(false);

  // Load states on mount
  useEffect(function () {
    getStates().then(function (res) {
      setStates(res.data);
      var total = res.data.reduce(function (sum, s) { return sum + s.count; }, 0);
      setTotalCount(total);
    }).catch(function (err) {
      console.error('Failed to load states:', err);
    });
  }, []);

  // Load data when filters change
  useEffect(function () {
    var params = { limit: 100 };
    if (selectedState) params.state = selectedState;
    if (selectedCounty) params.county = selectedCounty;
    if (searchQuery) params.search = searchQuery;

    setLoading(true);
    getDataCenters(params).then(function (res) {
      setDataCenters(res.data.dataCenters);
      setLoading(false);
    }).catch(function (err) {
      console.error('Failed to load data centers:', err);
      setLoading(false);
    });

    // Load stats
    var statsParams = {};
    if (selectedState) statsParams.state = selectedState;
    if (selectedCounty) statsParams.county = selectedCounty;
    getStats(statsParams).then(function (res) {
      setStats(res.data);
    });
  }, [selectedState, selectedCounty, searchQuery]);

  // Load counties when state changes
  useEffect(function () {
    if (selectedState) {
      getCounties(selectedState).then(function (res) {
        setCounties(res.data);
      });
    } else {
      setCounties([]);
    }
  }, [selectedState]);

  var handleStateChange = useCallback(function (state) {
    setSelectedState(state);
    setSelectedCounty('');
    setSelectedDC(null);
    setAnalysis(null);
  }, []);

  var handleCountyChange = useCallback(function (county) {
    setSelectedCounty(county);
    setSelectedDC(null);
    setAnalysis(null);
  }, []);

  var handleSearchChange = useCallback(function (query) {
    setSearchQuery(query);
  }, []);

var handleSelectDC = useCallback(function (dc) {
    if (!dc) {
      setSelectedDC(null);
      setAnalysis(null);
      return;
    }

    setSelectedDC(dc);
    setAnalysis(null);

    getDataCenterById(dc._id).then(function (res) {
      setSelectedDC(res.data.dataCenter);
      setAnalysis(res.data.analysis);
    }).catch(function (err) {
      console.error('Failed to load analysis:', err);
    });
  }, []);

  var handleLogout = useCallback(function () {
    auth.logout();
    if (auth0.isAuthenticated) {
      auth0.logout({ logoutParams: { returnTo: window.location.origin + '/login' } });
    } else {
      navigate('/login');
    }
  }, [auth, auth0, navigate]);

  return React.createElement('div', { className: 'dashboard' },
    React.createElement(MapView, {
      dataCenters: dataCenters,
      selectedDC: selectedDC,
      onSelectDC: handleSelectDC,
      analysis: analysis
    }),
    React.createElement(SidePanel, {
      states: states,
      counties: counties,
      dataCenters: dataCenters,
      selectedState: selectedState,
      selectedCounty: selectedCounty,
      searchQuery: searchQuery,
      stats: stats,
      selectedDC: selectedDC,
      analysis: analysis,
      onStateChange: handleStateChange,
      onCountyChange: handleCountyChange,
      onSearchChange: handleSearchChange,
      onSelectDC: handleSelectDC,
      onLogout: handleLogout,
      totalCount: totalCount
    })
  );
}

export default Dashboard;