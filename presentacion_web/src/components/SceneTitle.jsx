export default function SceneTitle({ number, section, children, className = '' }) {
  return <div className={`scene-heading reveal ${className}`}>
    <p className="eyebrow"><span className="scene-number">{number}</span><span className="eyebrow-divider">/</span>{section}</p>
    <h1 id="slide-title" tabIndex={-1}>{children}</h1>
  </div>;
}
