document.addEventListener('DOMContentLoaded', function() {
    const socket = io();
    const alertsContainer = document.getElementById('alerts-container');
    
    socket.on('new_alert', function(data) {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert-item ${data.status === 'success' ? 'success' : ''}`;
        
        const time = new Date(data.timestamp).toLocaleTimeString();
        const ipShort = data.source_ip.length > 15 ? 
            data.source_ip.substring(0, 15) + '...' : data.source_ip;
        
        alertDiv.innerHTML = `
            <div class="d-flex justify-content-between">
                <strong>${ipShort}</strong>
                <small class="text-muted">${time}</small>
            </div>
            <div>Status: <span class="badge bg-${data.status === 'success' ? 'success' : 'danger'}">
                ${data.status}
            </span></div>
        `;
        
        alertsContainer.insertBefore(alertDiv, alertsContainer.firstChild);
        
        // Limit to 20 alerts
        if (alertsContainer.children.length > 20) {
            alertsContainer.removeChild(alertsContainer.lastChild);
        }
    });
});