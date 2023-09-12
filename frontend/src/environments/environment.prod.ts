export const environment = {
  production: true,
  backendBaseUrl: "https://api.meldekopf.fski.ofd.itshessen.hessen.de",
  backendWSUrl: "https://api.meldekopf.fski.ofd.itshessen.hessen.de",
  keycloakUrl: "https://auth.fski-d.ofd.itshessen.hessen.de",
  keycloakRealm: "fski",
  keycloakClientId: "meldekopf-frontend",

  fahnderRightTemplate: { canRead: true, canWrite: false, role: "FahnderIn" },
  itRightTemplate: { canRead: true, canWrite: false, role: "IT" },
  objektleitungRightTemplate: { canRead: true, canWrite: true, role: "Objektleitung" },
  meldekopfRightTemplate: { canRead: true, canWrite: true, role: "Meldekopf" },
  federfuehrungRightTemplate: { canRead: true, canWrite: true, role: "Federfuerung" },
};
