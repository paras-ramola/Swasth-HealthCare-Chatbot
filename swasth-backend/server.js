require("dotenv").config();

const express = require("express");
const cors = require("cors");
const pool = require("./db");
const bcrypt = require("bcrypt");
const jwt = require("jsonwebtoken");
const chatRoutes = require("./routes/chatRoutes");

// JWT secret
const JWT_SECRET =
  process.env.JWT_SECRET || "swasth_dev_secret_change_in_prod";

const app = express();

// ---------------- MIDDLEWARE ----------------

app.use(
  cors({
    origin: "*", // change to your Vercel URL later for production
  })
);

app.use(express.json());
app.use("/api", chatRoutes);

// ---------------- ROOT ROUTE (fix for Render) ----------------
app.get("/", (req, res) => {
  res.json({
    status: "ok",
    message: "Swasth backend is running ",
  });
});

// ---------------- DATABASE TABLES ----------------
async function createTables() {
  try {
    await pool.query(`
      CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        email VARCHAR(255) UNIQUE NOT NULL,
        password VARCHAR(255) NOT NULL,
        age INT,
        gender VARCHAR(20),
        address TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      );
    `);

    await pool.query(`
      CREATE TABLE IF NOT EXISTS chat_history (
        id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
        symptoms TEXT,
        predicted_disease TEXT,
        is_emergency BOOLEAN DEFAULT false,
        emergency_message TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      );
    `);

    await pool.query(`
      CREATE TABLE IF NOT EXISTS chat_sessions (
        id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
        session_id VARCHAR(255) UNIQUE NOT NULL,
        collected_symptoms JSONB DEFAULT '[]',
        questions_asked INTEGER DEFAULT 0,
        conversation_history JSONB DEFAULT '[]',
        all_symptoms_ever JSONB DEFAULT '[]',
        mode VARCHAR(50) DEFAULT 'triage',
        current_prediction JSONB DEFAULT NULL,
        status VARCHAR(50) DEFAULT 'active',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      );
    `);

    await pool.query(`
      CREATE TABLE IF NOT EXISTS predictions (
        id SERIAL PRIMARY KEY,
        session_id VARCHAR(255),
        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
        symptoms JSONB,
        predicted_disease VARCHAR(255),
        confidence FLOAT,
        explanation TEXT,
        attempt_number INTEGER DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      );
    `);

    await pool.query(`
      CREATE TABLE IF NOT EXISTS prediction_feedback (
        id SERIAL PRIMARY KEY,
        prediction_id INTEGER REFERENCES predictions(id),
        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
        feedback_type VARCHAR(10),
        user_comment TEXT,
        llm_evaluation TEXT,
        user_was_correct BOOLEAN,
        resolution VARCHAR(50),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      );
    `);

    await pool.query(`
      CREATE TABLE IF NOT EXISTS assessments (
        id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
        session_id VARCHAR(255) UNIQUE NOT NULL,
        selected_symptoms JSONB DEFAULT '[]',
        confirmed_symptoms JSONB DEFAULT '[]',
        absent_symptoms JSONB DEFAULT '[]',
        asked_symptoms JSONB DEFAULT '[]',
        questions_asked INTEGER DEFAULT 0,
        status VARCHAR(50) DEFAULT 'in_progress',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      );
    `);

    await pool.query(`
      CREATE TABLE IF NOT EXISTS assessment_results (
        id SERIAL PRIMARY KEY,
        assessment_id INTEGER REFERENCES assessments(id),
        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
        predicted_disease VARCHAR(255),
        confidence FLOAT,
        all_predictions JSONB,
        explanation TEXT,
        feedback_type VARCHAR(10),
        feedback_comment TEXT,
        user_was_correct BOOLEAN,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      );
    `);

    console.log("All tables ready");
  } catch (err) {
    console.error("Table creation error:", err);
  }
}

createTables();

// ---------------- AUTH ----------------
app.post("/register", async (req, res) => {
  const { email, password, age, gender, address } = req.body;

  try {
    const hashedPassword = await bcrypt.hash(password, 10);

    await pool.query(
      "INSERT INTO users (email, password, age, gender, address) VALUES ($1, $2, $3, $4, $5)",
      [email, hashedPassword, age, gender, address]
    );

    res.json({ message: "User registered successfully" });
  } catch (err) {
    res.status(400).json({ error: "User already exists" });
  }
});

app.post("/login", async (req, res) => {
  const { email, password } = req.body;

  const result = await pool.query("SELECT * FROM users WHERE email = $1", [
    email,
  ]);

  if (result.rows.length === 0) {
    return res.status(400).json({ error: "User not found" });
  }

  const user = result.rows[0];
  const validPassword = await bcrypt.compare(password, user.password);

  if (!validPassword) {
    return res.status(400).json({ error: "Invalid password" });
  }

  const token = jwt.sign({ userId: user.id }, JWT_SECRET, {
    expiresIn: "7d",
  });

  res.json({ token });
});

// ---------------- PROFILE ----------------
const verifyToken = require("./middleware/authMiddleware");

app.get("/me", verifyToken, async (req, res) => {
  try {
    const userId = req.user?.userId;

    if (!userId) {
      return res.status(401).json({ error: "Unauthorized" });
    }

    const result = await pool.query(
      "SELECT id, email, age, gender, address FROM users WHERE id = $1",
      [userId]
    );

    if (result.rows.length === 0) {
      return res.status(404).json({ error: "User not found" });
    }

    const user = result.rows[0];

    res.json({
      id: user.id,
      email: user.email,
      fullName: user.email.split("@")[0],
      age: user.age,
      gender: user.gender,
      address: user.address,
    });
  } catch (err) {
    console.error("GET /me error:", err);
    res.status(500).json({ error: "Failed to fetch profile" });
  }
});

// ---------------- START SERVER (FIXED FOR RENDER) ----------------
const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});