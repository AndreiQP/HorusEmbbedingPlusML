const express = require('express');
const cors = require('cors'); // Necessário para permitir requisições da extensão do Chrome
const app = express();

app.use(cors());
app.use(express.json());

app.listen(3000, () => {
    console.log("🛡️ Servidor Backend IC Horus (Node proxy) rodando na porta 3000...");
});