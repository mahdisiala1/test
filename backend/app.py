from flask import Flask, request, jsonify
from flask_cors import CORS
import jwt
from datetime import datetime, timedelta
from functools import wraps
import os
import scrum_graph
from bson import ObjectId
import pymongo
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
app = Flask(__name__)
CORS(app, supports_credentials=True)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'man7ebbech')

# MongoDB setup (use MONGODB_URI env var for Atlas)
MONGO_URI = os.getenv("MONGODB_URI") or os.getenv("MONGO_URI") or "mongodb://localhost:27017"
client = pymongo.MongoClient(MONGO_URI)
# Prefer explicit MONGO_DB env; otherwise use DB embedded in the URI; fallback to 'scrum_db'
env_db = os.getenv('MONGO_DB')
if env_db:
    db = client[env_db]
else:
    db = client.get_default_database() or client['scrum_db']
users_col = db.get_collection('users')
plans_col = db.get_collection('plans')

# Helper: ensure demo user exists
if users_col.count_documents({}) == 0:
    demo_id = str(ObjectId())
    users_col.insert_one({
        'id': demo_id,
        'username': 'testuser',
        'email': 'test@example.com',
        'password': 'test123',  # demo only
        'created_at': datetime.utcnow().isoformat(),
    })

# Feature flag from scrum_graph (if present)
HAS_SCRUM_AGENT = getattr(scrum_graph, 'HAS_SCRUM_AGENT', True)


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        # Check if token is in headers
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]

        if not token:
            return jsonify({'error': 'Token is missing!'}), 401

        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            user_id = data.get('user_id')
            if not user_id:
                return jsonify({'error': 'Invalid token payload!'}), 401

            current_user = users_col.find_one({'id': user_id})
            if not current_user:
                return jsonify({'error': 'User not found!'}), 401

            # normalize id field
            current_user['id'] = current_user.get('id') or str(current_user.get('_id'))
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired!'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token!'}), 401

        return f(current_user, *args, **kwargs)

    return decorated


@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register new user"""
    try:
        data = request.get_json()

        # Check required fields
        if not data:
            return jsonify({"error": "No data provided"}), 400

        required_fields = ['username', 'email', 'password']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Check if user exists
        if users_col.find_one({'email': data['email']}) is not None:
            return jsonify({"error": "Email already registered"}), 400
        if users_col.find_one({'username': data['username']}) is not None:
            return jsonify({"error": "Username already taken"}), 400

        # Create new user (in production, hash the password!)
        new_id = str(ObjectId())
        user_doc = {
            'id': new_id,
            'username': data['username'],
            'email': data['email'],
            'password': data['password'],
            'created_at': datetime.utcnow().isoformat()
        }
        users_col.insert_one(user_doc)

        return jsonify({
            "success": True,
            "message": "User registered successfully",
            "user": {
                "id": new_id,
                "username": data['username'],
                "email": data['email']
            }
        }), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login user and return token"""
    try:
        data = request.get_json()

        if not data or 'email' not in data or 'password' not in data:
            return jsonify({"error": "Missing email or password"}), 400

        # Find user
        user = users_col.find_one({'email': data['email']})
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Check password (in production, use proper password hashing!)
        if user.get('password') != data['password']:
            return jsonify({"error": "Invalid password"}), 401

        # Generate JWT token
        token = jwt.encode({
            'user_id': user.get('id') or str(user.get('_id')),
            'username': user.get('username'),
            'exp': datetime.utcnow() + timedelta(hours=24)
        }, app.config['SECRET_KEY'])

        return jsonify({
            "success": True,
            "message": "Login successful",
            "token": token,
            "user": {
                "id": user.get('id') or str(user.get('_id')),
                "username": user.get('username'),
                "email": user.get('email')
            }
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/auth/me', methods=['GET'])
@token_required
def get_current_user(current_user):
    """Get current user info (protected route)"""
    return jsonify({
        "success": True,
        "user": {
            "id": current_user.get('id'),
            "username": current_user.get('username'),
            'email': current_user.get('email')
        }
    }), 200


@app.route('/api/auth/logout', methods=['POST'])
@token_required
def logout(current_user):
    """Logout user (in production, implement token blacklist)"""
    return jsonify({
        "success": True,
        "message": "Logged out successfully"
    }), 200


@app.route('/api/scrum/analyze', methods=['POST'])
@token_required
def analyze_spec(current_user):
    """Analyze a cahier de charge and return scrum planning result"""
    try:
        data = request.get_json() or {}

        cahier = data.get('documentContent') or data.get('cahier_de_charge')
        if not cahier:
            return jsonify({"error": "Missing documentContent / cahier_de_charge"}), 400

        team_members = data.get('teamMembers') or []
        sprint_duration = data.get('sprintDuration') or data.get('sprint_length_days') or 2

        # Build team object expected by scrum_graph
        team = {
            'sprint_length_days': int(sprint_duration),
            'sprint_capacity_points': data.get('sprintCapacityPoints', 20),
            'members': [
                {
                    'name': m.get('name') or m.get('display_name') or 'Member',
                    'skills': m.get('skills') or []
                } for m in team_members
            ]
        }

        # Run the planning graph. Prefer build_scrum_graph (safe), fall back to run_scrum_planning if present.
        if hasattr(scrum_graph, 'build_scrum_graph'):
            graph = scrum_graph.build_scrum_graph()
            result = graph.invoke({
                'cahier_de_charge': cahier,
                'team': team,
                'validation_attempts': 0,
                'max_validation_attempts': 1,
            })
        elif hasattr(scrum_graph, 'run_scrum_planning'):
            result = scrum_graph.run_scrum_planning(cahier, team)
        else:
            return jsonify({"error": "scrum_graph does not expose a planner function"}), 500

        return jsonify({"success": True, "plan": result}), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
