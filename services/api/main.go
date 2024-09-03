// Package main implements a lightweight Go HTTP API that exposes video
// understanding capabilities backed by TwelveLabs and AWS DynamoDB.
//
// Endpoints:
//   GET  /health                         liveness probe
//   GET  /search?q=<query>&limit=<n>     semantic video search via TwelveLabs
//   GET  /videos/{id}                    fetch video metadata from DynamoDB
//   GET  /videos                         list recent videos (paginated)
//
// Configuration is read from environment variables:
//   TWELVELABS_API_KEY   – TwelveLabs API key
//   TWELVELABS_INDEX_ID  – Index to search
//   DYNAMODB_TABLE       – DynamoDB table name
//   AWS_REGION           – AWS region (default: us-east-1)
//   PORT                 – HTTP port (default: 8081)
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/feature/dynamodb/attributevalue"
	"github.com/aws/aws-sdk-go-v2/service/dynamodb"
	"github.com/aws/aws-sdk-go-v2/service/dynamodb/types"
	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
)

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

type appConfig struct {
	TwelveLabsAPIKey  string
	TwelveLabsIndexID string
	DynamoDBTable     string
	AWSRegion         string
	Port              string
}

func loadConfig() appConfig {
	cfg := appConfig{
		TwelveLabsAPIKey:  mustEnv("TWELVELABS_API_KEY"),
		TwelveLabsIndexID: mustEnv("TWELVELABS_INDEX_ID"),
		DynamoDBTable:     getEnv("DYNAMODB_TABLE", "video-metadata"),
		AWSRegion:         getEnv("AWS_REGION", "us-east-1"),
		Port:              getEnv("PORT", "8081"),
	}
	return cfg
}

func mustEnv(key string) string {
	v := os.Getenv(key)
	if v == "" {
		slog.Error("required environment variable not set", "key", key)
		os.Exit(1)
	}
	return v
}

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

type apiError struct {
	Error   string `json:"error"`
	Code    int    `json:"code"`
	Message string `json:"message,omitempty"`
}

type healthResponse struct {
	Status    string `json:"status"`
	Timestamp string `json:"timestamp"`
	Version   string `json:"version"`
}

type searchClip struct {
	VideoID    string  `json:"video_id"`
	Score      float64 `json:"score"`
	Start      float64 `json:"start"`
	End        float64 `json:"end"`
	Confidence string  `json:"confidence"`
	ThumbnailURL string `json:"thumbnail_url,omitempty"`
}

type searchResponse struct {
	Query    string       `json:"query"`
	IndexID  string       `json:"index_id"`
	Results  []searchClip `json:"results"`
	Total    int          `json:"total"`
	PageInfo interface{}  `json:"page_info,omitempty"`
}

type videoRecord struct {
	VideoID         string            `json:"video_id"           dynamodbav:"video_id"`
	S3Key           string            `json:"s3_key"             dynamodbav:"s3_key"`
	S3URI           string            `json:"s3_uri"             dynamodbav:"s3_uri"`
	IndexID         string            `json:"index_id"           dynamodbav:"index_id"`
	Status          string            `json:"status"             dynamodbav:"status"`
	EmbeddingStatus string            `json:"embedding_status"   dynamodbav:"embedding_status"`
	SizeBytes       int64             `json:"size_bytes"         dynamodbav:"size_bytes"`
	SizeMB          float64           `json:"size_mb"            dynamodbav:"size_mb"`
	CreatedAt       string            `json:"created_at"         dynamodbav:"created_at"`
	IndexedAt       string            `json:"indexed_at,omitempty" dynamodbav:"indexed_at"`
	EmbeddedAt      string            `json:"embedded_at,omitempty" dynamodbav:"embedded_at"`
	EmbeddingS3     string            `json:"embedding_s3,omitempty" dynamodbav:"embedding_s3"`
	DurationSec     float64           `json:"duration_sec,omitempty" dynamodbav:"duration_sec"`
	Tags            map[string]string `json:"tags,omitempty"     dynamodbav:"tags"`
}

type videoListResponse struct {
	Items         []videoRecord `json:"items"`
	Count         int           `json:"count"`
	LastEvaluatedKey string     `json:"last_evaluated_key,omitempty"`
}

// ---------------------------------------------------------------------------
// Application
// ---------------------------------------------------------------------------

type app struct {
	cfg    appConfig
	db     *dynamodb.Client
	logger *slog.Logger
}

func newApp(cfg appConfig) (*app, error) {
	awsCfg, err := config.LoadDefaultConfig(
		context.Background(),
		config.WithRegion(cfg.AWSRegion),
	)
	if err != nil {
		return nil, fmt.Errorf("loading AWS config: %w", err)
	}

	return &app{
		cfg:    cfg,
		db:     dynamodb.NewFromConfig(awsCfg),
		logger: slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo})),
	}, nil
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

// handleHealth returns a simple liveness probe response.
func (a *app) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, healthResponse{
		Status:    "ok",
		Timestamp: time.Now().UTC().Format(time.RFC3339),
		Version:   "1.0.0",
	})
}

// handleSearch proxies semantic search to TwelveLabs and returns ranked clips.
func (a *app) handleSearch(w http.ResponseWriter, r *http.Request) {
	query := r.URL.Query().Get("q")
	if query == "" {
		writeError(w, http.StatusBadRequest, "missing required query parameter 'q'")
		return
	}

	limitStr := r.URL.Query().Get("limit")
	limit := 10
	if limitStr != "" {
		if l, err := strconv.Atoi(limitStr); err == nil && l > 0 && l <= 50 {
			limit = l
		}
	}

	a.logger.Info("search request", "query", query, "limit", limit)

	tlResp, err := a.callTwelveLabsSearch(r.Context(), query, limit)
	if err != nil {
		a.logger.Error("twelvelabs search failed", "err", err)
		writeError(w, http.StatusBadGateway, "upstream search service error")
		return
	}

	clips := parseTLSearchResponse(tlResp)
	writeJSON(w, http.StatusOK, searchResponse{
		Query:   query,
		IndexID: a.cfg.TwelveLabsIndexID,
		Results: clips,
		Total:   len(clips),
	})
}

// handleGetVideo fetches a single video's metadata from DynamoDB.
func (a *app) handleGetVideo(w http.ResponseWriter, r *http.Request) {
	videoID := chi.URLParam(r, "id")
	if videoID == "" {
		writeError(w, http.StatusBadRequest, "missing video id")
		return
	}

	result, err := a.db.GetItem(r.Context(), &dynamodb.GetItemInput{
		TableName: aws.String(a.cfg.DynamoDBTable),
		Key: map[string]types.AttributeValue{
			"video_id": &types.AttributeValueMemberS{Value: videoID},
		},
	})
	if err != nil {
		a.logger.Error("dynamodb get_item failed", "video_id", videoID, "err", err)
		writeError(w, http.StatusInternalServerError, "failed to fetch video metadata")
		return
	}
	if result.Item == nil {
		writeError(w, http.StatusNotFound, fmt.Sprintf("video '%s' not found", videoID))
		return
	}

	var rec videoRecord
	if err := attributevalue.UnmarshalMap(result.Item, &rec); err != nil {
		a.logger.Error("failed to unmarshal dynamodb item", "err", err)
		writeError(w, http.StatusInternalServerError, "data deserialization error")
		return
	}

	writeJSON(w, http.StatusOK, rec)
}

// handleListVideos returns a paginated list of recent video records from DynamoDB.
func (a *app) handleListVideos(w http.ResponseWriter, r *http.Request) {
	limitStr := r.URL.Query().Get("limit")
	limit := int32(20)
	if limitStr != "" {
		if l, err := strconv.Atoi(limitStr); err == nil && l > 0 && l <= 100 {
			limit = int32(l)
		}
	}

	scanInput := &dynamodb.ScanInput{
		TableName: aws.String(a.cfg.DynamoDBTable),
		Limit:     aws.Int32(limit),
	}

	// Support cursor-based pagination via last_key query param
	if lastKey := r.URL.Query().Get("last_key"); lastKey != "" {
		scanInput.ExclusiveStartKey = map[string]types.AttributeValue{
			"video_id": &types.AttributeValueMemberS{Value: lastKey},
		}
	}

	result, err := a.db.Scan(r.Context(), scanInput)
	if err != nil {
		a.logger.Error("dynamodb scan failed", "err", err)
		writeError(w, http.StatusInternalServerError, "failed to list videos")
		return
	}

	var records []videoRecord
	if err := attributevalue.UnmarshalListOfMaps(result.Items, &records); err != nil {
		a.logger.Error("failed to unmarshal scan results", "err", err)
		writeError(w, http.StatusInternalServerError, "data deserialization error")
		return
	}

	resp := videoListResponse{Items: records, Count: len(records)}
	if result.LastEvaluatedKey != nil {
		if v, ok := result.LastEvaluatedKey["video_id"].(*types.AttributeValueMemberS); ok {
			resp.LastEvaluatedKey = v.Value
		}
	}

	writeJSON(w, http.StatusOK, resp)
}

// ---------------------------------------------------------------------------
// TwelveLabs HTTP client (inline — no external dependency)
// ---------------------------------------------------------------------------

const tlBaseURL = "https://api.twelvelabs.io/v1.3"

func (a *app) callTwelveLabsSearch(ctx context.Context, query string, limit int) (map[string]interface{}, error) {
	payload := map[string]interface{}{
		"index_id":       a.cfg.TwelveLabsIndexID,
		"query":          query,
		"search_options": []string{"visual", "conversation"},
		"group_by":       "clip",
		"threshold":      "medium",
		"page_limit":     limit,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshaling request: %w", err)
	}

	reqURL := fmt.Sprintf("%s/search", tlBaseURL)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, reqURL, bytesReader(body))
	if err != nil {
		return nil, fmt.Errorf("building request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", a.cfg.TwelveLabsAPIKey)

	httpClient := &http.Client{Timeout: 30 * time.Second}
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("executing request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response: %w", err)
	}

	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("twelvelabs API returned %d: %s", resp.StatusCode, string(respBody))
	}

	var result map[string]interface{}
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("parsing response JSON: %w", err)
	}

	return result, nil
}

func parseTLSearchResponse(raw map[string]interface{}) []searchClip {
	data, ok := raw["data"].([]interface{})
	if !ok {
		return nil
	}

	clips := make([]searchClip, 0, len(data))
	for _, item := range data {
		m, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		clip := searchClip{
			VideoID:    getString(m, "video_id"),
			Score:      getFloat(m, "score"),
			Start:      getFloat(m, "start"),
			End:        getFloat(m, "end"),
			Confidence: getString(m, "confidence"),
		}
		if thumb, ok := m["thumbnail_url"].(string); ok {
			clip.ThumbnailURL = thumb
		}
		clips = append(clips, clip)
	}
	return clips
}

// ---------------------------------------------------------------------------
// Middleware
// ---------------------------------------------------------------------------

func requestLogger(logger *slog.Logger) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			start := time.Now()
			ww := middleware.NewWrapResponseWriter(w, r.ProtoMajor)
			next.ServeHTTP(ww, r)
			logger.Info("request",
				"method", r.Method,
				"path", r.URL.Path,
				"status", ww.Status(),
				"duration_ms", time.Since(start).Milliseconds(),
				"remote_addr", r.RemoteAddr,
			)
		})
	}
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

func (a *app) routes() http.Handler {
	r := chi.NewRouter()
	r.Use(middleware.Recoverer)
	r.Use(middleware.RequestID)
	r.Use(middleware.RealIP)
	r.Use(requestLogger(a.logger))
	r.Use(middleware.Timeout(30 * time.Second))

	r.Get("/health", a.handleHealth)
	r.Get("/search", a.handleSearch)
	r.Get("/videos", a.handleListVideos)
	r.Get("/videos/{id}", a.handleGetVideo)

	return r
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		slog.Error("failed to write JSON response", "err", err)
	}
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, apiError{Error: http.StatusText(status), Code: status, Message: msg})
}

func getString(m map[string]interface{}, key string) string {
	if v, ok := m[key].(string); ok {
		return v
	}
	return ""
}

func getFloat(m map[string]interface{}, key string) float64 {
	if v, ok := m[key].(float64); ok {
		return v
	}
	return 0
}

func bytesReader(b []byte) io.Reader {
	return &bytesReaderImpl{data: b}
}

type bytesReaderImpl struct {
	data   []byte
	offset int
}

func (r *bytesReaderImpl) Read(p []byte) (int, error) {
	if r.offset >= len(r.data) {
		return 0, io.EOF
	}
	n := copy(p, r.data[r.offset:])
	r.offset += n
	return n, nil
}

// Ensure url package is used (imported for potential query building).
var _ = url.QueryEscape

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

func main() {
	cfg := loadConfig()

	application, err := newApp(cfg)
	if err != nil {
		slog.Error("failed to initialise application", "err", err)
		os.Exit(1)
	}

	addr := ":" + cfg.Port
	application.logger.Info("starting video understanding API", "addr", addr)

	srv := &http.Server{
		Addr:         addr,
		Handler:      application.routes(),
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		application.logger.Error("server error", "err", err)
		os.Exit(1)
	}
}
