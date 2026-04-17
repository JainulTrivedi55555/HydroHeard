import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth0 } from '@auth0/auth0-react';
import { useAuth } from '../context/AuthContext';

function CallbackPage() {
  var navigate = useNavigate();
  var auth0 = useAuth0();
  var auth = useAuth();

  useEffect(function () {
    if (auth0.isLoading) return;

    if (auth0.isAuthenticated && auth0.user) {
      // Auth0 login successful - create or login user in our backend
      var email = auth0.user.email;
      var name = auth0.user.name || auth0.user.nickname || 'Auth0 User';

      // Try login first, if fails then register
      auth.login(email, 'auth0_' + auth0.user.sub).then(function () {
        navigate('/dashboard');
      }).catch(function () {
        auth.register(name, email, 'auth0_' + auth0.user.sub).then(function () {
          navigate('/dashboard');
        }).catch(function () {
          navigate('/dashboard');
        });
      });
    } else if (!auth0.isLoading && !auth0.isAuthenticated) {
      navigate('/login');
    }
  }, [auth0.isLoading, auth0.isAuthenticated, auth0.user, auth, navigate]);

  return (
    <div style={{
      height: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: '#0a1a10',
      color: '#38bdf8',
      fontFamily: "'Space Mono', monospace",
      letterSpacing: '3px',
      fontSize: '14px'
    }}>
      AUTHENTICATING WITH AUTH0...
    </div>
  );
}

export default CallbackPage;