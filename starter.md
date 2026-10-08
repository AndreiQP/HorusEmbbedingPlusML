# Starter — API Horus no cluster e extensão local

Este guia inicia a API BGE/Transformer em uma GPU do cluster Recod, cria uma ponte SSH
até o Windows e conecta a extensão local ao WhatsApp Web.

## 1. Pré-requisitos

- repositório clonado localmente e no cluster;
- chave SSH cadastrada no cluster;
- ambiente Conda `env_gpu_final` disponível no cluster;
- BGE-M3 já presente no cache local do Hugging Face;
- checkpoint finalista da seed 42 em `backend/experiment_results/transformer/`;
- Chrome ou Edge com modo de desenvolvedor habilitado.

O fluxo de rede é:

```text
Extensão no navegador
    ↓ http://127.0.0.1:5500
porta local do Windows
    ↓ túnel SSH usando o alias headnode
headnode do cluster
    ↓ rede interna
nó GPU:5000
    ↓
API Flask + BGE-M3 + Transformer seed 42
```

## 2. Configurar o SSH no Windows

Exemplo para `%USERPROFILE%\.ssh\config`:

```sshconfig
Host recod
    HostName ssh.recod.ic.unicamp.br
    User SEU_USUARIO
    IdentityFile ~/.ssh/id_rsa_recod
    IdentitiesOnly yes

Host headnode
    HostName headnode
    User SEU_USUARIO
    ProxyCommand ssh recod -W %h:%p
    IdentityFile ~/.ssh/id_rsa_recod
    IdentitiesOnly yes
```

Teste:

```powershell
ssh headnode
```

Use `headnode` nos túneis. O alias `recod` alcança apenas o gateway público e não deve
ser usado como destino final para encaminhar portas até os nós GPU.

## 3. Atualizar o projeto no cluster

No PowerShell:

```powershell
ssh headnode
```

No headnode:

```bash
cd /home/SEU_USUARIO/HorusEmbbedingPlusML
git switch main
git pull --ff-only origin main
```

Não descarte arquivos modificados no cluster. Se o pull informar sobre alterações que
seriam sobrescritas, revise e preserve esses arquivos antes de continuar.

## 4. Reservar uma GPU

Ainda no headnode:

```bash
srun --partition=l40s \
  --gres=gpu:1 \
  --time=04:00:00 \
  --pty bash
```

Já dentro da alocação:

```bash
hostname
```

Anote o resultado, por exemplo `dl-03`. O nó pode mudar a cada alocação e será usado no
túnel do passo 6.

Ative o ambiente:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_gpu_final
cd /home/SEU_USUARIO/HorusEmbbedingPlusML
```

Confira a GPU e as bibliotecas:

```bash
nvidia-smi

python - <<'PY'
import torch
import sentence_transformers

print("Torch:", torch.__version__)
print("CUDA disponível:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Sentence Transformers:", sentence_transformers.__version__)
PY
```

Faça essa verificação dentro do nó alocado, não no headnode.

## 5. Iniciar a API

No nó GPU:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HORUS_DEVICE=auto

python -u -m backend.api.main
```

Saída esperada:

```text
Horus BGE API iniciada na porta 5000 (device=cuda)
Running on http://127.0.0.1:5000
```

Mantenha esse terminal aberto. `HORUS_DEVICE=auto` tenta CUDA e usa CPU se necessário.
Para exigir a tentativa de GPU, use `HORUS_DEVICE=cuda`; a implementação ainda fará
fallback para CPU quando CUDA não estiver disponível ou falhar.

Após um `git pull`, sempre reinicie esse processo para carregar o código novo.

## 6. Abrir o túnel no Windows

Abra outro PowerShell e substitua `dl-03` pelo `hostname` obtido no passo 4:

```powershell
ssh -N -o ExitOnForwardFailure=yes `
  -L 127.0.0.1:5500:dl-03:5000 `
  headnode
```

O comando permanece aberto e sem prompt enquanto o túnel estiver ativo. Isso é normal.
Mantenha esse PowerShell aberto.

Não use `192.168.x.x` por meio do alias `recod`: essa rota não alcança necessariamente
o nó de execução. Use o nome do nó e o alias `headnode`.

## 7. Testar a ponte

Em um terceiro PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:5500/api/health
```

Resultado esperado:

```text
status modelLoaded device
------ ----------- ------
ok            True cuda
```

Se `device` for `cpu`, a API está funcional, mas sem aceleração GPU.

## 8. Carregar e configurar a extensão

1. Abra `chrome://extensions`.
2. Ative o modo de desenvolvedor.
3. Clique em **Carregar sem compactação**.
4. Selecione a pasta local `horus_extension`.
5. Abra **Detalhes → Opções da extensão**.
6. Salve `http://127.0.0.1:5500` como URL da API.
7. Recarregue a extensão e depois recarregue `https://web.whatsapp.com`.

Se a extensão já estava instalada, recarregue-a após cada atualização de
`manifest.json`, `content.js`, `background.js` ou `annotation_utils.js`.

## 9. Executar uma análise

1. Clique explicitamente na conversa pela lista lateral do WhatsApp.
2. Abra o painel Horus pelo botão `👁️`.
3. Aguarde um gatilho lexical ou clique em **Forçar IA**.
4. Observe a classificação e a probabilidade no painel.
5. Para Scam, aguarde o job de explicabilidade:
   - balões pró-Scam recebem contorno vermelho;
   - o trecho pró-Scam mais forte recebe realce;
   - ao chegar mensagem nova, as marcas anteriores ficam esmaecidas.

O terminal da API deverá registrar `POST /api/analyse_history` e consultas a
`GET /api/explanations/<job_id>`.

## 10. Encerrar

- API: `Ctrl+C` no terminal do nó GPU;
- alocação interativa: `exit`;
- túnel SSH: `Ctrl+C` no PowerShell do túnel.

## Solução de problemas

### O túnel fica aberto, mas a API não responde

- confirme que o túnel usa `headnode`, não `recod`;
- confirme o nome atual do nó com `hostname`;
- no headnode, teste `getent hosts NOME_DO_NO`;
- confirme que a API continua aberta no nó GPU.

### A extensão não gera requisições

- teste `/api/health` pelo PowerShell;
- confira nas opções da extensão `http://127.0.0.1:5500`;
- recarregue a extensão e a aba do WhatsApp;
- clique novamente na conversa pela lista lateral;
- inspecione o service worker em `chrome://extensions` para ver a URL do erro.

### Hugging Face tenta acessar a internet

Confirme:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

O BGE precisa já existir no cache do usuário do cluster.

### `CUDA error: uncorrectable ECC error`

É uma falha do dispositivo/nó, não do payload. Encerre a alocação e solicite outra GPU.
Use `sacct -j ID_DO_JOB --format=JobID,State,ExitCode,NodeList%30` para identificar o
nó problemático.

### A API usa código antigo depois do pull

O processo Python não é atualizado em execução. Encerre com `Ctrl+C` e execute
novamente `python -u -m backend.api.main`.

### As mensagens estão contornadas, mas o trecho não está realçado

O front só realça quando o texto e os offsets retornados correspondem sem ambiguidade ao
DOM atual. O contorno é mantido como fallback seguro. Confira também se API e extensão
estão na mesma revisão do Git.

## Documentação relacionada

- [Extensão](horus_extension/README.md)
- [API](backend/api/README.md)
- [Plano da explicabilidade online](backend/machine_learning/studies/ONLINE_EXPLAINABILITY_PLAN.md)
