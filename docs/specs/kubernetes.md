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
- 24/11/2026, bloco 1 (kind v0.33.0, Kubernetes v1.37.0, um nó):
  - `artifacts/onnx` foi para o `.dockerignore` (a API não o referencia). A camada `COPY artifacts/`
    caiu de 546 MB para 152 kB; o `docker image ls` foi de 3,33 para 2,29 GB (com o armazenamento do
    containerd, ele soma a camada compactada e a descompactada). A camada do BERT na imagem do
    cluster tem 436 MB; a imagem dentro do nó, 907 MB compactados.
  - O `kind load docker-image` grava um `.tar` temporário no Windows, e o disco do PC não tinha
    espaço. A imagem foi carregada com `docker save` direto no `ctr images import` do nó, sem arquivo.
  - Sem probes, o pod fica `Ready` 2 s depois de criado e o rollout "termina" na hora, mas a API só
    abre a porta 12 s depois de o contêiner iniciar (carga do SVM, spaCy e BERT do disco). Nesse
    intervalo o Service mandaria tráfego para um pod que não responde. Isso é o bloco 2.
  - Memória do contêiner da API em repouso, com o BERT, depois de 4 requisições: 527 MB (`crictl
    stats`). O README diz "~2 GB"; falta medir sob carga (bloco 2).
  - `/api/analyze`: 110 a 270 ms por frase pelo port-forward, 4 chamadas (só verificação).
  - Achado anterior a este projeto: o log avisa `InconsistentVersionWarning`, porque o
    `ensemble.joblib` foi salvo com scikit-learn 1.6.1 e a imagem usa 1.9.1 (fixado desde set/2026).
    Fica registrado; não é corrigido aqui.
  - Sem `metrics-server` ainda (`kubectl top` falha): entra no bloco 3, com o HPA.
- 24/11/2026, disco: o C: do PC chegou a 0 GB com as imagens; o Hugo mudou o disco do Docker Desktop
  para o E:. O `kind load docker-image` de 3 GB levou 487 s, e isso pesa em cada versão nova.
- 24/11/2026, bloco 2:
  - **`/api/live` e `/api/ready`** (4 testes novos, 29 no total). A porta só abre depois da carga
    dos modelos, então só "estar pronto" não separaria as duas rotas. A separação vem de
    `REQUIRE_BERT` (padrão 0, o Docker e o CI de hoje não mudam): com 1 e sem BERT, `/api/ready` dá
    503 e `/api/live` dá 200. As duas são `async`, para não esperar na fila de threads do
    `/api/analyze`; o risco de timeout com rotas síncronas não foi medido.
  - **Prova no cluster:** uma versão com `ENABLE_BERT=0` e `REQUIRE_BERT=1` ficou viva (0
    reinícios), fora do Service (`ready=false` no EndpointSlice) e com `/api/ready` 503
    `missing: ["bert"]`. O rollout travou e o pod antigo seguiu atendendo. Desfeito com
    `kubectl rollout undo`.
  - **ConfigMap** pelo `configMapGenerator`: o nome leva o hash do conteúdo, então mudar um valor
    gera um rolling update.
  - **SIGTERM:** com `sh -c "uvicorn ..."`, o PID 1 era o `sh` e o uvicorn não recebia o sinal: o pod
    levou 32 s para sair (prazo de 30 s e SIGKILL), sem "Shutting down" no log. Com `exec`, o
    uvicorn é o PID 1: o pod sai em 9 s (5 s de `preStop` mais o encerramento), com "Shutting
    down" até "Finished server process". `terminationGracePeriodSeconds: 20`.
  - **Subida com as probes:** 18 s do agendamento até `Ready`, com o pod sozinho no nó (5 s até o
    contêiner iniciar, 6 s de import, 6 s de carga dos modelos). Com o pod antigo ainda no nó, 43 s.
  - **Recursos**, 300 requisições com 10 simultâneas, pelo port-forward, uma rodada de cada:

    | | Vazão | p50 / p95 (cliente) | CPU (pico) | Memória (pico) | Erros |
    |---|---|---|---|---|---|
    | Repouso | | | ~0 | 437 MiB | |
    | Sem limites | 4,0 req/s | 2,47 / 3,88 s | 3,6 núcleos | 619 MiB | 0 |
    | Limit de 2 núcleos | 4,1 req/s | 2,38 / 3,72 s | 2,06 núcleos | 570 MiB | 0 |

    Requests: 1 núcleo e 512 Mi. Limits: 2 núcleos e 1 Gi. Acima de 2 núcleos não houve ganho de
    vazão (hipótese: requisições simultâneas, cada uma com 2 threads do torch, disputam a CPU), por
    isso mais vazão vem de mais réplicas. Com o control plane (950m), o nó reserva 1.950m de 4.000m:
    cabem 3 réplicas. Nenhuma probe falhou sob carga (eventos `Unhealthy` só na subida, no
    encerramento e no teste sem BERT).
- 25/11/2026, bloco 3:
  - **Carga de dentro do cluster** (`k8s/carga/job.yaml`, um Job com a imagem da API e o
    `bench/load_test.py` num ConfigMap): o `port-forward` prende um pod e cai quando ele sai, então
    os erros seriam do túnel, não do Service. O `load_test.py` passou a contar os erros por tipo.
  - **Rolling update sob carga** (600 requisições, 10 simultâneas, `kubectl rollout restart` 20 s
    depois do início; `maxSurge: 1`, `maxUnavailable: 0`):

    | Configuração | Rodadas | Rollout | Erros |
    |---|---|---|---|
    | Probes + `preStop` (bloco 2) | 2 | 30 s e 20 s | 0 e 0 |
    | Sem probes e sem `preStop` (como o bloco 1) | 1 | 7 s | 472, todos `ConnectError` |

    Sem probes, o pod novo ficou pronto na hora e o antigo saiu com o novo ainda carregando os
    modelos. O número 472 está inflado: a recusa de conexão é instantânea e o teste não espera entre
    tentativas, então os 10 clientes gastaram a fila nessa janela. Ele prova a queda, não mede a
    duração. A configuração foi restaurada com `kubectl apply -k` (as probes voltaram).
  - **metrics-server v0.8.0** (`k8s/metrics-server`, manifesto oficial e `--kubelet-insecure-tls`,
    porque o certificado do kubelet no kind é autoassinado).
  - **HPA** de 1 a 3 réplicas, 70% do request de CPU, `scaleDown` com janela de 60 s (o padrão é
    300). O `replicas` saiu do Deployment para o `apply` não brigar com o HPA.
  - **Correção do registro do bloco 2:** "cabem 3 réplicas" estava errado; faltava o metrics-server.
    O sistema reserva 1.050m, e 3 réplicas de 1 núcleo passam dos 4.000m: a terceira ficou
    `Pending` (`0/1 nodes are available: 1 Insufficient cpu`). O request não foi baixado só para a
    demonstração caber.
  - **Escala** (1.500 requisições, 10 simultâneas): 34 s depois do início, a CPU média estava em 167%
    e o HPA pediu 3 de uma vez; a segunda réplica ficou pronta em ~24 s; a terceira, `Pending`. A
    carga acabou em 315 s, a métrica zerou em ~33 s e a escala voltou a 1 em 407 s (~90 s depois
    do fim: atraso da métrica mais a janela de 60 s). 0 erros, inclusive na volta para 1 réplica.
  - **Vazão, uma rodada de cada, mesma carga:**

    | | Vazão | p50 / p95 (cliente) | CPU da API |
    |---|---|---|---|
    | 1 pod (HPA com máximo 1) | 4,6 req/s | 2,11 / 3,15 s | ~2.000m (no limit) |
    | HPA, 2 pods prontos | 5,0 req/s | 1,91 / 3,10 s | ~1.000m por pod |

    O HPA funcionou, mas a vazão quase não mudou (+9%, dentro do que uma rodada só permite dizer).
    Com 2 pods, a API usou os mesmos ~2 núcleos de 1 pod; o pod de carga usou 15 a 19m. Ainda não
    sei o que segurou o resto do nó de 4 núcleos (i3-9100F, 4 núcleos físicos). Hipóteses não
    testadas: a CPU do PC disputada pelo Windows e pelo Docker Desktop (a do host não foi medida);
    o teste de carga com 10 clientes fixos, que limita a vazão a ~10 / latência.
  - **Investigação da vazão** (mesma escala do HPA, 1.500 requisições, CPU lida a cada 10 s no pod,
    no nó e no Windows; requisições por pod contadas no log):

    | | Com keep-alive | Sem keep-alive (`--no-keepalive`) |
    |---|---|---|
    | Requisições no pod criado pelo HPA | 0 | 591 de 1.500 |
    | CPU da API depois da escala | 2.000m num pod, 2m no novo | 1.400 a 1.800m em cada |
    | CPU do nó | ~2,2 núcleos | ~3,5 núcleos |
    | CPU do Windows | 55 a 78% | 84 a 100% |
    | Vazão; p50 / p95 | 5,8 req/s; 1,72 / 2,45 s | 8,0 req/s; 1,19 / 2,26 s |
    | Erros | 0 | 0 |

    **Causa:** o Service (kube-proxy, camada 4) escolhe o pod quando a conexão abre. As 10 conexões
    keep-alive do teste abriram com 1 pod e ficaram nele; o pod novo ficou pronto e ocioso. A média
    de 100% que o HPA mostrava era um pod a 200% e outro a 0%. Sem keep-alive, o tráfego se dividiu e
    o gargalo passou a ser o i3 (Windows a 100%). As duas hipóteses anteriores (CPU do host, clientes
    fixos) não explicavam o pod novo parado e ficam descartadas como causa principal.
    **Ressalva:** a vazão de 1 pod variou de 4,6 a 5,8 req/s entre rodadas, então o ganho firme é o
    tráfego chegar ao pod novo e a API passar de ~2 para ~3,5 núcleos, não a porcentagem exata.
    **Fora do escopo:** balanceamento por requisição (um Ingress ou proxy de camada 7, ou o servidor
    fechar conexões antigas). Fica como próximo passo no README.
- 25/11/2026, bloco 4:
  - **Overlay `k8s/ci`:** a mesma base, imagem `misogyny-lens:ci` (o Dockerfile normal, só SVM) e
    `REQUIRE_BERT=0`. Os recursos não mudaram: a imagem só com SVM usa 360 a 374 MiB, e o runner de
    4 núcleos comporta o pod e a réplica extra da troca de versão. O "menos recursos" da decisão 2
    não foi preciso.
  - **`k8s/ci/testar.sh`** (o CI chama; rodado no PC antes do push): deploy, teste de fumaça pelo
    Service (um pod com `kubectl run`) e troca de versão sob 20.000 requisições, exigindo 0 erros.
    Se a carga acabar antes da troca, o teste falha como inconclusivo.
  - **Rodada local:** deploy pronto em 15 s; fumaça ok (`verdict_model: svm`); troca de versão em
    11 s com 0 erros em 20.000 requisições (158 req/s).
  - **CI:** job `kubernetes` com `helm/kind-action@v1.12.0` (kind v0.33.0). O tempo no GitHub só
    existe depois do push.
  - **README:** seção do Kubernetes com os números; os "~2 GB" viraram 437 a 619 MiB medidos; o
    tamanho da imagem passou a ser o da camada (1,65 GB do `pip install`), porque o `docker image ls`
    soma a camada compactada e a descompactada.
