import React, { useEffect, useRef } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import './MapView.css';

function MapView({ dataCenters, selectedDC, onSelectDC, analysis }) {
  const mapRef = useRef(null);
  const mapInstance = useRef(null);
  const markersRef = useRef(null);
  const selectedMarkerRef = useRef(null);
  const selectedCircleRef = useRef(null);

  useEffect(function initMap() {
    if (!mapInstance.current) {
      mapInstance.current = L.map(mapRef.current, { zoomControl: true }).setView([37, -96], 4);
      L.tileLayer('https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', {
        maxZoom: 21,
        attribution: 'Google Satellite'
      }).addTo(mapInstance.current);
    }

    return function cleanup() {
      if (mapInstance.current) {
        mapInstance.current.remove();
        mapInstance.current = null;
      }
    };
  }, []);

  useEffect(function updateMarkers() {
    if (!mapInstance.current) return;

    if (markersRef.current) {
      mapInstance.current.removeLayer(markersRef.current);
    }

    if (!dataCenters || dataCenters.length === 0) return;

    var group = L.featureGroup();

    dataCenters.forEach(function (dc) {
      var sz = dc.sqft > 100000 ? 14 : 8;
      var color = dc.sqft > 100000 ? '#22c55e' : '#60a5fa';
      var icon = L.divIcon({
        className: '',
        html: '<div style="width:' + sz + 'px;height:' + sz + 'px;background:' + color + ';border:2px solid #fff;border-radius:50%;box-shadow:0 1px 6px rgba(0,0,0,0.4);cursor:pointer"></div>',
        iconSize: [sz, sz],
        iconAnchor: [sz / 2, sz / 2]
      });

      var marker = L.marker([dc.lat, dc.lon], { icon: icon });
      marker.on('click', function () { onSelectDC(dc); });
      marker.addTo(group);
    });

    group.addTo(mapInstance.current);
    mapInstance.current.fitBounds(group.getBounds().pad(0.15));
    markersRef.current = group;
  }, [dataCenters, onSelectDC]);

  useEffect(function handleSelection() {
    if (!mapInstance.current) return;

    if (selectedMarkerRef.current) {
      mapInstance.current.removeLayer(selectedMarkerRef.current);
      selectedMarkerRef.current = null;
    }
    if (selectedCircleRef.current) {
      mapInstance.current.removeLayer(selectedCircleRef.current);
      selectedCircleRef.current = null;
    }

    if (!selectedDC || !analysis) return;

    mapInstance.current.flyTo([selectedDC.lat, selectedDC.lon], 17, { duration: 1.2 });

    var radiusM = Math.sqrt(selectedDC.sqft / Math.PI) * 0.3048;
    selectedCircleRef.current = L.circle([selectedDC.lat, selectedDC.lon], {
      radius: radiusM,
      color: '#0ea5e9',
      weight: 2,
      fillColor: '#0ea5e9',
      fillOpacity: 0.15,
      dashArray: '6 4'
    }).addTo(mapInstance.current);

    var icon = L.divIcon({
      className: '',
      html: '<div style="width:36px;height:36px;background:#0ea5e9;border:3px solid #fff;border-radius:50%;box-shadow:0 3px 15px rgba(0,0,0,0.5);display:flex;align-items:center;justify-content:center;font-size:14px;color:#fff;font-weight:bold">W</div>',
      iconSize: [36, 36],
      iconAnchor: [18, 18]
    });

    selectedMarkerRef.current = L.marker([selectedDC.lat, selectedDC.lon], { icon: icon }).addTo(mapInstance.current);

    var popupContent =
      '<div style="font-family:Outfit,sans-serif;font-size:12px;min-width:220px;line-height:1.5">' +
      '<b style="font-size:13px">' + selectedDC.name + '</b><br>' +
      '<span style="color:#666">' + (selectedDC.operator || 'Unknown') + '</span><br>' +
      '<b>' + selectedDC.sqft.toLocaleString() + '</b> sqft roof<br>' +
      'Harvest: <b>' + (analysis.femp.annualHarvest / 1000).toFixed(0) + 'K</b> gal/yr<br>' +
      'Savings: <b>$' + analysis.femp.annualSavings.toLocaleString() + '</b>/yr<br>' +
      'Viability: <b>' + analysis.viability.total + '/100</b><br>' +
      '<a href="https://www.google.com/maps?q=' + selectedDC.lat + ',' + selectedDC.lon + '&z=18&t=k" target="_blank" style="color:#0ea5e9">Open in Google Maps</a>' +
      '</div>';

    selectedMarkerRef.current.bindPopup(popupContent).openPopup();
  }, [selectedDC, analysis]);

  var overlayContent = null;
  if (selectedDC && analysis) {
    var mapsUrl = 'https://www.google.com/maps?q=' + selectedDC.lat + ',' + selectedDC.lon + '&z=18&t=k';
    var harvestText = (analysis.femp.annualHarvest / 1000).toFixed(0) + 'K gal/yr';
    var savingsText = '$' + analysis.femp.annualSavings.toLocaleString() + '/yr';
    var scoreText = 'Score: ' + analysis.viability.total + '/100';
    var subText = selectedDC.county + ', ' + selectedDC.state;
    if (selectedDC.operator) {
      subText = subText + ' - ' + selectedDC.operator;
    }

    overlayContent = React.createElement('div', { className: 'map-overlay' },
      React.createElement('div', { className: 'overlay-top' },
        React.createElement('div', { className: 'overlay-name' }, selectedDC.name),
        React.createElement('a', { className: 'overlay-btn', href: mapsUrl, target: '_blank', rel: 'noreferrer' }, 'Earth View')
      ),
      React.createElement('div', { className: 'overlay-sub' }, subText),
      React.createElement('div', { className: 'overlay-stats' },
        React.createElement('span', { style: { color: '#38bdf8' } }, selectedDC.sqft.toLocaleString() + ' sqft'),
        React.createElement('span', { style: { color: '#4ade80' } }, harvestText),
        React.createElement('span', { style: { color: '#fbbf24' } }, savingsText),
        React.createElement('span', { style: { color: '#a78bfa' } }, scoreText)
      )
    );
  }

  return (
    <div className="map-container">
      <div ref={mapRef} className="map"></div>
      <div className="map-logo">
        <svg width="22" height="22" viewBox="0 0 32 32" fill="none">
          <path d="M16 3C16 3 6 15 6 21C6 26.5 10.5 29 16 29C21.5 29 26 26.5 26 21C26 15 16 3 16 3Z" fill="#0ea5e9" opacity=".3" stroke="#0ea5e9" strokeWidth="2"></path>
        </svg>
        <div>
          <span className="logo-text">RAINUSE NEXUS</span>
          <span className="logo-sub">BY GRUNDFOS</span>
        </div>
      </div>
      <div className="map-chips">
        <div className="ai-chip">
          <span className="chip-dot green"></span>GEMINI AI SCORING
        </div>
        <div className="ai-chip purple">
          <span className="chip-dot purple-dot"></span>SOLANA VERIFIED
        </div>
      </div>
      {overlayContent}
    </div>
  );
}

export default MapView;