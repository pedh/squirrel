//
//  LlmSuggest.swift
//  Squirrel
//
//  Async LLM-based candidate suggestion for Squirrel input method.
//  Shows local candidates first, then appends LLM-recommended candidates at the top.
//

import Foundation

final class LlmSuggest {
  struct Config {
    var enabled = false
    var endpoint = "http://localhost:8000/v1/chat/completions"
    var model = "qwen3.6-max-preview"
    var timeout: TimeInterval = 0.5
    var maxCandidates = 3

    static func fromSquirrelConfig(_ config: SquirrelConfig?) -> Config {
      var cfg = Config()
      guard let config = config else { return cfg }
      cfg.enabled = config.getBool("llm_suggest/enabled") ?? false
      if let endpoint = config.getString("llm_suggest/endpoint") {
        cfg.endpoint = endpoint
      }
      if let model = config.getString("llm_suggest/model") {
        cfg.model = model
      }
      if let timeout = config.getDouble("llm_suggest/timeout") {
        cfg.timeout = timeout
      }
      if let max = config.getDouble("llm_suggest/max_candidates") {
        cfg.maxCandidates = Int(max)
      }
      return cfg
    }
  }

  private var config: Config
  private var currentTask: URLSessionDataTask?
  private var cachedResults: [String: [String]] = [:]
  private var consecutiveFailures = 0
  private let maxConsecutiveFailures = 5
  private var disabled = false

  init(config: Config) {
    self.config = config
  }

  func updateConfig(_ newConfig: Config) {
    config = newConfig
    disabled = false
    consecutiveFailures = 0
    cachedResults.removeAll()
  }

  /// Fetch LLM suggestions asynchronously. Calls the callback on main thread.
  /// - Parameters:
  ///   - preedit: Current pinyin input string
  ///   - localCandidates: Local candidate list from librime
  ///   - context: Previously committed text (conversation context)
  ///   - callback: Called with merged candidate list (LLM candidates prepended, deduplicated)
  func fetch(preedit: String, localCandidates: [String], context: String,
             callback: @escaping ([String], [String]) -> Void) {
    if !config.enabled || disabled || preedit.isEmpty {
      return
    }

    // Cancel previous request (debounce)
    currentTask?.cancel()
    currentTask = nil

    // Check cache
    if let cached = cachedResults[preedit] {
      let (merged, comments) = merge(llmCandidates: cached, into: localCandidates)
      callback(merged, comments)
      return
    }

    let systemPrompt = """
    你是输入法候选词排序助手。根据对话历史和当前输入，选出最符合语境的词。\
    按优先级从高到低列出最多\(config.maxCandidates)个候选词，用顿号分隔，不要拼音、序号或解释。
    """

    var userPrompt = ""
    if !context.isEmpty {
      userPrompt += "上文：" + context + "\n"
    }
    userPrompt += "当前输入：" + preedit + "\n"
    userPrompt += "本地候选：" + localCandidates.joined(separator: "、")

    let requestBody: [String: Any] = [
      "model": config.model,
      "messages": [
        ["role": "system", "content": systemPrompt],
        ["role": "user", "content": userPrompt]
      ],
      "max_tokens": 20,
      "temperature": 0.0
    ]

    guard let url = URL(string: config.endpoint) else { return }
    guard let httpBody = try? JSONSerialization.data(withJSONObject: requestBody) else { return }

    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.httpBody = httpBody
    request.timeoutInterval = config.timeout

    let task = URLSession.shared.dataTask(with: request) { [weak self] data, _, error in
      guard let self = self else { return }

      if error != nil {
        self.handleFailure()
        return
      }

      guard let data = data else {
        self.handleFailure()
        return
      }

      let llmCandidates = self.parseResponse(data)
      if llmCandidates.isEmpty {
        self.handleFailure()
        return
      }

      self.consecutiveFailures = 0
      self.disabled = false
      self.cachedResults[preedit] = llmCandidates

      let (merged, comments) = self.merge(llmCandidates: llmCandidates, into: localCandidates)
      DispatchQueue.main.async {
        callback(merged, comments)
      }
    }
    currentTask = task
    task.resume()
  }

  private func parseResponse(_ data: Data) -> [String] {
    guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let choices = json["choices"] as? [[String: Any]],
          let firstChoice = choices.first,
          let message = firstChoice["message"] as? [String: Any],
          let content = message["content"] as? String else {
      return []
    }
    // Parse "word1、word2、word3" format, also handle comma and space separators
    let candidates = content
      .replacingOccurrences(of: "、", with: ",")
      .replacingOccurrences(of: "，", with: ",")
      .split(separator: ",")
      .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
      .filter { !$0.isEmpty && $0.count <= 10 }  // filter out overly long or empty results
    return Array(candidates.prefix(config.maxCandidates))
  }

  /// Merge LLM candidates into local candidate list.
  /// If an LLM candidate already exists in local list, move it to the front instead of duplicating.
  private func merge(llmCandidates: [String], into localCandidates: [String]) -> ([String], [String]) {
    var merged = localCandidates
    var comments = merged.map { _ in "" }  // empty comments for local candidates

    // Track which local candidates were moved to front
    var movedIndices = Set<Int>()

    for llmCand in llmCandidates.reversed() {  // reverse so first LLM candidate ends up at position 0
      if let existingIndex = merged.firstIndex(of: llmCand) {
        // Move existing candidate to front
        let existingComment = comments[existingIndex]
        merged.remove(at: existingIndex)
        comments.remove(at: existingIndex)
        merged.insert(llmCand, at: 0)
        comments.insert("🤖", at: 0)  // mark as LLM-suggested even if it was a local candidate
      } else {
        // Insert new candidate at front
        merged.insert(llmCand, at: 0)
        comments.insert("🤖", at: 0)
      }
    }

    return (merged, comments)
  }

  private func handleFailure() {
    consecutiveFailures += 1
    if consecutiveFailures >= maxConsecutiveFailures {
      disabled = true
      print("LlmSuggest: disabled after \(maxConsecutiveFailures) consecutive failures")
    }
  }

  func clearCache() {
    cachedResults.removeAll()
  }
}
