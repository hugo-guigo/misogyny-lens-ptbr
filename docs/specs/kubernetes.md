# Especificação: a API em Kubernetes local

Escrita antes do código (24/11/2026). O registro da revisão fica no fim.

## Objetivo

Rodar a API (ensemble SVM + BERTimbau) num cluster Kubernetes local, com probes, recursos medidos,
rolling update sem queda comprovado por carga, HPA se o PC aguentar e encerramento gracioso. Um CI
sobe o mesmo deploy num cluster efêmero. É um cluster local de estudo: nada de nuvem, nada de
"produção" nem "escala".

## O que a leitura do repositório mostrou

- **Uma rota de saúde para tudo:** `/api/health` responde o status e se o BERT carregou. Não separa
  "o processo está vivo" de "o modelo está pronto".
- **O modelo carrega antes de a porta abrir.** O uvicorn só aceita conexões depois do `lifespan`,
  que carrega SVM, spaCy e BERT, e o BERT pode baixar 436 MB do Hugging Face na subida. Até lá, nenhuma
  rota responde, nem a de saúde. Sem uma `startupProbe`, a `livenessProbe` mataria o pod no meio da
  carga.
- **A imagem carrega 521 MB que a API não usa:** `artifacts/onnx` (o modelo do navegador) não está no
  `.dockerignore`, ao contrário de `artifacts/bert`. Bloco 1: medir a imagem antes e depois de tirar.
- **PID 1 é o `sh`:** o `CMD` é `sh -c "uvicorn ..."`. Se o `sh` não repassar o SIGTERM, o Kubernetes
  espera o prazo e mata o processo no meio das requisições. Vou conferir na prática antes de mudar.
- **O README diz "~2 GB de RAM" para o ensemble.** Ainda não está medido no contêiner.
- **O PC:** o Docker Desktop tem 4 CPUs e 7,7 GB. O `kubectl` 1.36 já está instalado, o kind não.

## Decisões (com a alternativa recusada)

1. **kind, não k3d.** É o projeto oficial da comunidade Kubernetes (SIG Testing), roda um Kubernetes
   padrão dentro de um contêiner e é a ferramenta comum de CI (a action `helm/kind-action`). Assim, a
   mesma ferramenta roda no PC e no CI. O k3d (k3s) é mais leve e já traz o metrics-server, mas é uma
   distribuição enxuta, com componentes trocados (Traefik, SQLite no lugar do etcd), e o que eu
   aprendesse nele valeria menos para um cluster padrão. Instalação: `winget install Kubernetes.kind`,
   grátis.
2. **Kustomize (`kubectl apply -k`), não Helm.** É um serviço só, e o Kustomize já vem no `kubectl`.
   Um overlay `ci/` troca o que muda no CI (imagem só com SVM, menos recursos). Helm traria templates
   e um chart para manter sem ganho aqui.
3. **BERT dentro da imagem do cluster local**, num alvo separado do Dockerfile, para a imagem de hoje
   não mudar. Recusados:
   - baixar na subida de cada pod: 436 MB por réplica a cada rolling update e a cada réplica nova do
     HPA, numa conexão doméstica;
   - um `Job` que baixa uma vez num `PersistentVolumeClaim`: ensina mais conceitos, mas não cabe em 1 a
     2 dias junto com o resto. Fica registrado como próximo passo.

   O CI usa a imagem só com SVM (sem baixar o BERT), como o CI de hoje, e o README diz isso.
4. **Três probes, com rotas separadas:**
   - `startupProbe` em `/api/live`, com prazo largo para a carga do modelo;
   - `livenessProbe` em `/api/live`: só "o processo responde";
   - `readinessProbe` em `/api/ready`: 200 só com os modelos carregados.

   `/api/health` continua igual, porque o CI e o README de hoje usam essa rota.
5. **Requests e limits medidos.** Primeiro roda sem limite e mede com `kubectl top` e `docker stats`
   (em repouso e sob carga). Depois fixa requests perto do repouso e limits com folga do pico. Os
   números vão para o README.
6. **Encerramento gracioso.**
   - Conferir se o SIGTERM chega ao uvicorn. Se não chegar, trocar `sh -c` por `exec`.
   - `preStop` com uma pausa curta, para o Service tirar o pod da lista antes de o processo parar de
     aceitar conexões.
   - `terminationGracePeriodSeconds` maior que a requisição mais lenta.
7. **Rolling update** com `maxUnavailable: 0` e `maxSurge: 1`. A prova é o `bench/load_test.py` rodando
   durante a troca de versão, contando os erros. A troca é por uma anotação no pod template, que gera
   um ReplicaSet novo sem mudar o código.
8. **HPA com metrics-server**, se o PC aguentar duas ou três réplicas com BERT (cada uma deve passar de
   1 GB). Mostra as réplicas subindo sob carga e descendo depois. Se não couber, fica documentado com os
   números que mostram por quê.
9. **Acesso:** `kubectl port-forward` para o teste de carga. Sem Ingress, que não acrescenta nada com um
   serviço só.

## Blocos (com a sua revisão entre eles)

1. Instalar o kind, criar o cluster, imagem (com e sem `artifacts/onnx`, medida), carregar no cluster,
   Deployment e Service no ar.
2. `/api/live` e `/api/ready` com testes, as três probes, requests e limits medidos, ConfigMap
   (`ENABLE_BERT`, `TORCH_THREADS`) e SIGTERM conferido.
3. Rolling update sob carga, com a contagem de erros, e HPA (se couber).
4. CI com kind (imagem SVM, rollout e teste de fumaça) e README com os números.

Cada bloco passa por `/code-review` em low antes do commit. `docs/DECISOES.md` fica fora do git, com
as decisões e a seção "o que acontece quando dou `kubectl apply`".

## Riscos

- **Memória:** com várias réplicas com BERT, o Docker Desktop (7,7 GB) pode não aguentar o HPA. Se
  for o caso, fica documentado com números.
- **Tempo de subida do pod:** carregar o BERT na CPU do i3 pode levar dezenas de segundos, e isso
  alonga o rolling update. Também fica medido e documentado.
- **CI:** imagem grande (torch na CPU) e cluster kind no runner do GitHub. O tempo do job fica no
  README.

## Registro da revisão

- 24/11/2026, revisão do Hugo: spec aprovada, com a instalação do kind (v0.33.0, via winget).
