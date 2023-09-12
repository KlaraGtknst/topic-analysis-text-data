export const environment = {
  production: false,
  backendBaseUrl: "http://localhost:3000",
  backendWSUrl: "http://localhost:3000",
  keycloakUrl: "http://localhost:8080",
  keycloakRealm: "fski",
  keycloakClientId: "meldekopf-frontend",

  fahnderRightTemplate: { canRead: true, canWrite: false, role: "FahnderIn" },
  itRightTemplate: { canRead: true, canWrite: false, role: "IT" },
  objektleitungRightTemplate: { canRead: true, canWrite: true, role: "Objektleitung" },
  meldekopfRightTemplate: { canRead: true, canWrite: true, role: "Meldekopf" },
  federfuehrungRightTemplate: { canRead: true, canWrite: true, role: "Federfuerung" },
};
